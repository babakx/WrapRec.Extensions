using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WrapRec.Core;

namespace WrapRec.Extensions.Models
{
    public enum NegSampler
    {
        CombinedSampling,
        OnlyObserved,
        OnlyUnobserved
    }

	public enum UnobservedSamplingMethod
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
		public Dictionary<string, List<int>> UserFeedbackLevels { get; private set; }
		public List<Feedback> TrainFeedback { get; private set; }
		public List<string> AllItems { get; private set; }
        public List<Feedback> AllPosFeedback { get; set; }
		public bool IsRatingBased { get; set; }
        public int[] Levels { get; set; }
        public int NumObservedNeg { get; set; }
        public int NumUnobservedNeg { get; set; }
        public NegSampler NegSampler { get; set;}
		public UnobservedSamplingMethod UnobservedSamplingMethod { get; set; }

        protected override void InitModel()
		{
			base.InitModel();
            
            // make sure the train and test items are specified in the Container
            Split.UpdateFeedbackSlices();
            TrainFeedback = Split.Train.ToList();
            Levels = Split.Train.Select(f => f.Level).Distinct().OrderByDescending(l => l).ToArray();

            // index user pos and negative feedback 
			UserFeedback = new Dictionary<string, List<Feedback>>();
			UserPosFeedback = new Dictionary<string, List<Feedback>>();
			UserNegFeedback = new Dictionary<string, List<Feedback>>();
			UserFeedbackLevels = new Dictionary<string, List<int>>();
            AllPosFeedback = new List<Feedback>();
			AllItems = Split.Train.Select(f => f.Item.Id).Distinct().ToList();

			var usersFeedback = Split.Train.GroupBy(f => f.User);
			if (IsRatingBased)
			{
				foreach (var g in usersFeedback)
				{
					float ratingAvg = g.Average(f => ((Rating)f).Value);
                    string userId = g.Key.Id;
					UserFeedback[userId] = new List<Feedback>();
					UserPosFeedback[userId] = new List<Feedback>();
                    UserNegFeedback[userId] = new List<Feedback>();
					UserFeedbackLevels[userId] = g.Select(f => f.Level).Distinct().OrderByDescending(l => l).ToList();

                    foreach (Rating f in g)
                    {
						UserFeedback[userId].Add(f);
						
						if (f.Value > ratingAvg)
                        {
                            UserPosFeedback[userId].Add(f);
                            AllPosFeedback.Add(f);
                        }
                        else
                            UserNegFeedback[userId].Add(f);
                    }
				}
			}
			else
			{
				UserPosFeedback = usersFeedback.ToDictionary(g => g.Key.Id, g => g.ToList());
				UserNegFeedback = usersFeedback.ToDictionary(g => g.Key.Id, g => new List<Feedback>());
				UserFeedback = UserPosFeedback;
				AllPosFeedback = TrainFeedback;
				UserFeedbackLevels = UserFeedback.ToDictionary(g => g.Key, 
					g => g.Value.Select(f => f.Level).Distinct().OrderByDescending(l => l).ToList());
			}
		}

        protected virtual Feedback SamplePosFeedback()
        {
            if (UniformUserSampling)
            {
                string userIdOrg = UsersMap.ToOriginalID(SampleUser());
                var userPosFeedback = UserPosFeedback[userIdOrg];
                return userPosFeedback[random.Next(userPosFeedback.Count)];
            }
            else
            {
                return AllPosFeedback[random.Next(AllPosFeedback.Count)];
            }
        }

		protected virtual Feedback SampleObservedNegFeedback(Feedback posFeedback)
		{
			return SampleObservedNegFeedback(posFeedback, -1);
		}

        protected virtual Feedback SampleObservedNegFeedback(Feedback posFeedback, int sampleLevel = -1)
        {
			if (sampleLevel == -1)
			{
				var toSampleLevels = GetNegSampleLevels(posFeedback);
				
				// in case of no observed negative feedback, sampler fallbacks to unobserved feedback
				if (toSampleLevels.Count == 0)
					return SampleUnobservedNegFeedback(posFeedback);

				sampleLevel = toSampleLevels[random.Next(toSampleLevels.Count)];
			}

			NumObservedNeg++;
			var toSample = UserPosFeedback[posFeedback.User.Id].Concat(UserNegFeedback[posFeedback.User.Id])
				.Where(f => f.Level == sampleLevel).ToList();
			return toSample[random.Next(toSample.Count)];
        }

		public virtual Feedback SampleUnobservedNegFeedback(Feedback posFeedback)
		{
			Feedback neg = null;
			
			switch (UnobservedSamplingMethod)
			{
				case UnobservedSamplingMethod.UniformFeedback:
					do
					{
						neg = TrainFeedback[random.Next(TrainFeedback.Count)];
					} while (neg.User == posFeedback.User);
					break;
				case UnobservedSamplingMethod.UniformItem:
					string itemId;
					do
					{
						itemId = AllItems[random.Next(AllItems.Count)];
					} while (UserFeedback[posFeedback.User.Id].Select(f => f.Item.Id).Contains(itemId));
					neg = new Feedback(posFeedback.User, Split.Container.Items[itemId]);
					break;
				case UnobservedSamplingMethod.Dynamic:
					throw new NotImplementedException();
				default:
					break;
			}
			
			NumUnobservedNeg++;
			return neg;
		}

		protected virtual Feedback SampleNegFeedback(Feedback posFeedback)
		{
			var toSampleLevels = GetNegSampleLevels(posFeedback);
			int sampleLevelIndex = random.Next(toSampleLevels.Count + 1);

			// in case the index overflows the list of levels or there is no sample levels (both values below are 0), sample unobserved
			if (sampleLevelIndex == toSampleLevels.Count)
				return SampleUnobservedNegFeedback(posFeedback);
			
			return SampleObservedNegFeedback(posFeedback, toSampleLevels[sampleLevelIndex]);
		}

		protected virtual List<int> GetNegSampleLevels(Feedback posFeedback)
		{
			return UserFeedbackLevels[posFeedback.User.Id]
				.Where(l => l < posFeedback.Level)
				.ToList();
		}

		// this method ingores the property of the baseClass: WithReplacement
		public override void Iterate()
		{
			Func<Feedback, Feedback> negSampler = SampleNegFeedback;

			switch (NegSampler)
			{
				case NegSampler.CombinedSampling:
					negSampler = SampleNegFeedback;
					break;
				case NegSampler.OnlyObserved:
					negSampler = SampleObservedNegFeedback;
					break;
				case NegSampler.OnlyUnobserved:
					negSampler = SampleUnobservedNegFeedback;
					break;
			}

			for (int i = 0; i < Feedback.Count; i++)
			{
				var pos = SamplePosFeedback();
				var neg = negSampler(pos);

				int user_id = UsersMap.ToInternalID(pos.User.Id);
				int item_id = ItemsMap.ToInternalID(pos.Item.Id);
				int other_item_id = ItemsMap.ToInternalID(neg.Item.Id);

				UpdateFactors(user_id, item_id, other_item_id, true, true, update_j);
			}
		}

		public override void Train()
		{
			Logger.Current.Info("Initializing MultiLevel BPR...");
			InitModel();
	
			Logger.Current.Info("Training with MultiLevel BPR...");
			random = MyMediaLite.Random.GetInstance();
            for (int i = 0; i < NumIter; i++)
                Iterate();

            Logger.Current.Info("Num unobserved negative samples: {0}, observed negative samples: {1}", 
				NumObservedNeg, NumUnobservedNeg);
		}

	}
}
