using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WrapRec.Core;
using MathNet.Numerics.Distributions;
using LinqLib.Sequence;
using LinqLib.Operators;
using WrapRec.Utils;

namespace WrapRec.Extensions.Models
{
	public enum PosSampler
	{ 
		UniformUser,
		UniformLevel,
		DynamicLevel,
		UniformFeedback
	}
	
	public enum UnobservedNegSampler
	{ 
		UniformFeedback,
		UniformItem,
		Dynamic
	}

    public class MultiLevelBPRFM : BPRFM
	{
		public Dictionary<string, List<Feedback>> UserFeedback { get; private set; }
		public Dictionary<string, List<Feedback>> UserPosFeedback { get; private set; }
        public Dictionary<string, List<Feedback>> UserNegFeedback { get; private set; }
		public Dictionary<int, List<Feedback>> LevelPosFeedback { get; set; }
		public Dictionary<string, List<int>> UserFeedbackLevels { get; private set; }
		public MultiKeyDictionary<string, int, List<Feedback>> UserLevelFeedback { get; private set; }
		public List<Feedback> TrainFeedback { get; private set; }
		public List<string> AllItems { get; private set; }
        public List<Feedback> AllPosFeedback { get; set; }
        public int[] Levels { get; set; }
		public List<int> PosLevels { get; private set; }
        public int NumObservedNeg { get; set; }
        public int NumUnobservedNeg { get; set; }
		public PosSampler PosSampler { get; set; }
		public UnobservedNegSampler UnobservedNegSampler { get; set; }
		public float Lambda { get; set; }
		public float LambdaLevel { get; set; }
		public float UnobservedRatio { get; set; }

		Categorical _rankSampler;
		Dictionary<int, List<string>> _factorBasedRank;
		Dictionary<int, int> _levelIndex;
		float[] _itemFactorsStdev;
		Categorical _posLevelSampler;
		Categorical _unobservedOrNegativeSampler;

		public MultiLevelBPRFM()
			: base()
		{
			Lambda = 500f;
			LambdaLevel = 10f;
		}

        protected override void InitModel()
		{
			base.InitModel();
			CacheSplitData();

			// initialize dynamic sampler
			_factorBasedRank = new Dictionary<int, List<string>>();

			double[] rankingPro = new double[AllItems.Count];
			double sum = 0;
			for (int i = 0; i < AllItems.Count; i++)
			{
				rankingPro[i] = Math.Exp(-(i + 1.0) / Lambda);
				sum += rankingPro[i];
			}
			for (int i = 0; i < AllItems.Count; i++)
				rankingPro[i] /= sum;

			_rankSampler = new Categorical(rankingPro);
			_itemFactorsStdev = new float[NumFactors];

			// initialize dynamic level sampler
			if (PosSampler == PosSampler.DynamicLevel)
			{
				// key is levelNo, value id list of feedback in that level
				sum = LevelPosFeedback.Sum(kv => kv.Key * kv.Value.Count);
				double[] levelPro = new double[PosLevels.Count];

				for (int i = 0; i < PosLevels.Count; i++)
					levelPro[i] = 1.0f * PosLevels[i] * LevelPosFeedback[PosLevels[i]].Count / sum;
				
				_posLevelSampler = new Categorical(levelPro);
			}

			// this sampler specifies whether the negative sample should be sampled from observed or unobserved feedback
			// the parameter of this distributio is parameter beta in Loni_RecSys2016
			// with beta=1 always unobserved will be sampled but with beta = 0 always observed will be sampled (if possible)
			_unobservedOrNegativeSampler = new Categorical(new double[] { (1 - UnobservedRatio), UnobservedRatio });
		}

		protected virtual void CacheSplitData()
		{
			// make sure the train and test items are specified in the Container
			Split.UpdateFeedbackSlices();
			TrainFeedback = Split.Train.ToList();
			Levels = Split.Train.Select(f => f.Level).Distinct().OrderByDescending(l => l).ToArray();
			PosLevels = new List<int>();

			UserFeedback = new Dictionary<string, List<Feedback>>();
			UserPosFeedback = new Dictionary<string, List<Feedback>>();
			UserNegFeedback = new Dictionary<string, List<Feedback>>();
			UserFeedbackLevels = new Dictionary<string, List<int>>();
			UserLevelFeedback = new MultiKeyDictionary<string, int, List<Core.Feedback>>();
			AllPosFeedback = new List<Feedback>();
			AllItems = Split.Train.Select(f => f.Item.Id).Distinct().ToList();

			LevelPosFeedback = new Dictionary<int, List<Feedback>>();

			var usersFeedback = Split.Train.GroupBy(f => f.User);
			foreach (var g in usersFeedback)
			{
				string userId = g.Key.Id;
				UserFeedback[userId] = new List<Feedback>();
				UserPosFeedback[userId] = new List<Feedback>();
				UserNegFeedback[userId] = new List<Feedback>();
				UserFeedbackLevels[userId] = g.Select(f => f.Level).Distinct().OrderByDescending(l => l).ToList();
				
				float ratingAvg = -1f;
				if (g.First() is Rating)
					ratingAvg = g.Average(f => ((Rating)f).Value);

				foreach (Feedback f in g)
				{
					UserFeedback[f.User.Id].Add(f);

					if (!UserLevelFeedback.ContainsKey(f.User.Id, f.Level))
						UserLevelFeedback.Add(f.User.Id,f.Level, new List<Feedback>());

					UserLevelFeedback[f.User.Id][f.Level].Add(f);

					// determine whether the feedback is positive, negative or rating and add that to the right list
					switch (f.FeedbackType)
					{
						case FeedbackType.Positive:
							UserPosFeedback[f.User.Id].Add(f);
							AllPosFeedback.Add(f);
							
							if (!LevelPosFeedback.ContainsKey(f.Level))
							{
								LevelPosFeedback[f.Level] = new List<Feedback>();
								PosLevels.Add(f.Level);
							}
							LevelPosFeedback[f.Level].Add(f);
							break;
						case FeedbackType.Negative:
							UserNegFeedback[f.User.Id].Add(f);
							break;
						case FeedbackType.Rating:
							if (((Rating)f).Value >= ratingAvg)
							{
								UserPosFeedback[f.User.Id].Add(f);
								AllPosFeedback.Add(f);
								if (!LevelPosFeedback.ContainsKey(f.Level))
								{
									LevelPosFeedback[f.Level] = new List<Feedback>();
									PosLevels.Add(f.Level);
								}
								LevelPosFeedback[f.Level].Add(f);
							}
							else
								UserNegFeedback[f.User.Id].Add(f);
							break;
						default:
							break;
					}
				}
			}
			
			PosLevels = PosLevels.OrderByDescending(l => l).ToList();
			_levelIndex = new Dictionary<int, int>();

			for (int i = 0; i < PosLevels.Count; i++)
				_levelIndex.Add(PosLevels[i], i);

			Logger.Current.Info("Number of feedbacks:");
			LevelPosFeedback.Select(kv => new { Level = kv.Key, Count = kv.Value.Count })
				.OrderByDescending(l => l.Level)
				.Select(l => string.Format("Level {0}: {1}", l.Level, l.Count))
				.ForEach(l => Logger.Current.Info(l));
			Logger.Current.Info("");
		}

        protected virtual Feedback SamplePosFeedback()
        {
			switch (PosSampler)
			{
				case PosSampler.UniformUser:
					string userIdOrg = UsersMap.ToOriginalID(SampleUser());
					var userPosFeedback = UserPosFeedback[userIdOrg];
					return userPosFeedback[random.Next(userPosFeedback.Count)];
				case PosSampler.UniformLevel:
					int level = PosLevels[random.Next(PosLevels.Count)];
					// uniform feedback sampling from a level
					int index = random.Next(LevelPosFeedback[level].Count);
					return LevelPosFeedback[level][index];
				case PosSampler.DynamicLevel:
					int l = PosLevels[_posLevelSampler.Sample()];
					int i = random.Next(LevelPosFeedback[l].Count);
					return LevelPosFeedback[l][i];
				case PosSampler.UniformFeedback:
					return AllPosFeedback[random.Next(AllPosFeedback.Count)];
				default:
					return null;
			}
        }


		public virtual Feedback SampleUnobservedNegFeedback(Feedback posFeedback)
		{
			Feedback neg = null;
			
			switch (UnobservedNegSampler)
			{
				case UnobservedNegSampler.UniformFeedback:
					do
					{
						neg = TrainFeedback[random.Next(TrainFeedback.Count)];
					} while (neg.User == posFeedback.User);
					break;
				case UnobservedNegSampler.UniformItem:
					string itemId;
					do
					{
						itemId = AllItems[random.Next(AllItems.Count)];
					} while (UserFeedback[posFeedback.User.Id].Select(f => f.Item.Id).Contains(itemId));
					neg = new Feedback(posFeedback.User, Split.Container.Items[itemId]);
					break;
				case UnobservedNegSampler.Dynamic:
					string negItemId;
					do
					{
						negItemId = SampleNegItemDynamic(posFeedback);
					} while (UserFeedback[posFeedback.User.Id].Select(f => f.Item.Id).Contains(negItemId));
					neg = new Feedback(posFeedback.User, Split.Container.Items[negItemId]);
					break;
				default:
					break;
			}
			
			NumUnobservedNeg++;
			return neg;
		}

		protected virtual Feedback SampleNegFeedback(Feedback posFeedback)
		{
			int observedOrUnobserved = _unobservedOrNegativeSampler.Sample();
			if (observedOrUnobserved == 1)
				return SampleUnobservedNegFeedback(posFeedback);

			var toSampleLevels = GetNegSampleLevels(posFeedback);
			// not possible to sample observed
			if (toSampleLevels.Count == 0)
				return SampleUnobservedNegFeedback(posFeedback);
			
			// sample observed
			// here both levels and item are sampled uniformly with respect to the frequency of items in a level
			var cdf = new List<int>();
			cdf.Add(0);

			for (int i = 0; i < toSampleLevels.Count; i++)
				cdf.Add(cdf[i] + UserLevelFeedback[posFeedback.User.Id][toSampleLevels[i]].Count);

			int sampleIndex = random.Next(cdf.Last());

			int levelIndex = -1, sampleOffset = -1;
			for (int i = 1; i < cdf.Count; i++)
			{
				if (sampleIndex < cdf[i])
				{
					sampleOffset = sampleIndex - cdf[i - 1];
					levelIndex = i - 1;
					break;
				}
			}

			NumObservedNeg++;
			return UserLevelFeedback[posFeedback.User.Id][toSampleLevels[levelIndex]][sampleOffset];
		}

		/// <summary>
		/// sample negative level non-uniforly with respect to the frequency items in the level and importance of level
		/// </summary>
		/// <param name="allowedLevels">The levels that user has items in</param>
		/// <returns></returns>
		protected virtual int SampleNegativeLevel(List<int> allowedLevels)
		{
			int sum = 0;
			foreach (int level in allowedLevels)
			{
				sum += LevelPosFeedback[level].Count * level;
			}

			double[] probs = new double[allowedLevels.Count + 1];
			for (int i = 0; i < allowedLevels.Count; i++)
			{
				probs[i] = (1 - UnobservedRatio) * allowedLevels[i] * LevelPosFeedback[allowedLevels[i]].Count / sum;
			}
			probs[allowedLevels.Count] = UnobservedRatio;

			int levelIndex = new Categorical(probs).Sample();

			if (levelIndex == allowedLevels.Count)
				return 0;

			return allowedLevels[levelIndex];
		}

		protected virtual List<int> GetNegSampleLevels(Feedback posFeedback)
		{
			return UserFeedbackLevels[posFeedback.User.Id]
				.Where(l => l < posFeedback.Level)
				.ToList();
		}

		// this method ingores the properties of the baseClass: WithReplacement and UniformUserSampling
		public override void Iterate()
		{
			for (int i = 0; i < Feedback.Count; i++)
			{
				if (UnobservedNegSampler == UnobservedNegSampler.Dynamic && i % (AllItems.Count * Math.Log(AllItems.Count)) == 0)
					UpdateDynamicSampler();
				
				var pos = SamplePosFeedback();
				var neg = SampleNegFeedback(pos);

				int user_id = UsersMap.ToInternalID(pos.User.Id);
				int item_id = ItemsMap.ToInternalID(pos.Item.Id);
				int other_item_id = ItemsMap.ToInternalID(neg.Item.Id);

				UpdateFactors(user_id, item_id, other_item_id, true, true, update_j);
			}
		}

		protected virtual void UpdateDynamicSampler()
		{
 			for (int f = 0; f < NumFactors; f++)
			{
				_factorBasedRank[f] = AllItems.OrderByDescending(iId => Math.Abs(item_factors[ItemsMap.ToInternalID(iId), f])).ToList();
				_itemFactorsStdev[f] = AllItems.Select(iId => item_factors[ItemsMap.ToInternalID(iId), f]).Stdev();
			}
		}

		
		protected virtual string SampleNegItemDynamic(Feedback posFeedback)
		{
			// sample r
			int r;

			do
			{
				r = _rankSampler.Sample();
			} while (r >= AllItems.Count);

			int user_id = UsersMap.ToInternalID(posFeedback.User.Id);
			var u = user_factors.GetRow(user_id);

			// sample f from p(f|c)
			double sum = 0;
			for (int i = 0; i < NumFactors; i++)
				sum += Math.Abs(u[i]) * _itemFactorsStdev[i];

			double[] probs = new double[NumFactors];
			for (int i = 0; i < NumFactors; i++)
				probs[i] = Math.Abs(u[i]) * _itemFactorsStdev[i] / sum;

			int f = new Categorical(probs).Sample();

			// take item j (negItemId) from position r of the list of sampled f
			string negItemId;
			if (Math.Sign(user_factors[user_id, f]) > 0)
				negItemId = _factorBasedRank[f][r];
			else
				negItemId = _factorBasedRank[f][AllItems.Count - r - 1];

			return negItemId;
		}

		public override void Train()
		{
			Logger.Current.Info("Initializing MultiLevel BPR...");
			InitModel();
	
			Logger.Current.Info("Training with MultiLevel BPR...");
			random = MyMediaLite.Random.GetInstance();
			for (int i = 0; i < NumIter; i++)
			{
				Iterate();
				float perc = ((float)(i + 1) / NumIter) * 100;
				Console.Write("\r{0:0}%   ", perc);
			}
            Logger.Current.Info("\nNum Observed negative samples: {0}, Unobserved negative samples: {1}", 
				NumObservedNeg, NumUnobservedNeg);
		}

	}
}
