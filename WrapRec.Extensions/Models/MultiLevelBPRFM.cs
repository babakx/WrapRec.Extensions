using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WrapRec.Core;

namespace WrapRec.Extensions.Models
{
	public class MultiLevelBPRFM : BPRFM
	{
		int[] _feedbackLevels;
		int _numUnobservedSamples = 0;
		int _numObservedSamples = 0;
		Dictionary<string, List<Feedback>> _trainUserPosFeedback;
		Dictionary<string, List<Feedback>> _trainUserNegFeedback;
		Dictionary<string, List<string>> _trainUserPosItems;
		List<Feedback> _trainFeedback;

		int _numTrainSamples;

		public bool IsRatingBased { get; set; }
		public bool OverSamplePopular { get; set; }

		protected override void InitModel()
		{
			base.InitModel();
			Split.UpdateFeedbackSlices();
			_feedbackLevels = Split.Train.Select(f => f.Level).Distinct().ToArray();
			
			_trainUserPosFeedback = new Dictionary<string, List<Feedback>>();
			_trainUserNegFeedback = new Dictionary<string, List<Core.Feedback>>();
			_trainUserPosItems = new Dictionary<string, List<string>>();
			_trainFeedback = Split.Train.ToList();
			_numTrainSamples = _trainFeedback.Count;

			var usersFeedback = Split.Train.GroupBy(f => f.User);

			if (IsRatingBased)
			{
				foreach (var g in usersFeedback)
				{
					float ratingAvg = g.Average(f => ((Rating)f).Value);
					_trainUserPosFeedback.Add(g.Key.Id, g.Where(f => ((Rating)f).Value > ratingAvg).ToList());
					_trainUserNegFeedback.Add(g.Key.Id, g.Where(f => ((Rating)f).Value <= ratingAvg).ToList());
					_trainUserPosItems.Add(g.Key.Id, g.Where(f => ((Rating)f).Value > ratingAvg).Select(f => f.Item.Id).ToList());
				}
			}
			else
			{
				_trainUserPosFeedback = usersFeedback.ToDictionary(g => g.Key.Id, g => g.ToList());
				_trainUserNegFeedback = usersFeedback.ToDictionary(g => g.Key.Id, g => new List<Feedback>());
			}
		}

		public override void Iterate()
		{
			if (!OverSamplePopular)
			{
				base.Iterate();
				return;
			}

			for (int i = 0; i < Feedback.Count; i++)
			{
				int user_id = SampleUser();
				string userIdOrg = UsersMap.ToOriginalID(user_id);
				var userPosFeedback = _trainUserPosFeedback[userIdOrg];
				var userPosItems = _trainUserPosItems[userIdOrg];

				var samplePosFeedback = userPosFeedback[random.Next(userPosFeedback.Count)];
				int item_id = ItemsMap.ToInternalID(samplePosFeedback.Item.Id);

				string negItemIdOrg; 
				do
				{
					negItemIdOrg = _trainFeedback[random.Next(_numTrainSamples)].Item.Id;
				} while (userPosItems.Contains(negItemIdOrg));

				int negItemId = ItemsMap.ToInternalID(negItemIdOrg);

				UpdateFactors(user_id, item_id, negItemId, true, true, update_j);
			}
		}

		protected override void SampleTriple(out int user_id, out int item_id, out int other_item_id)
		{
			user_id = SampleUser();
			string userIdOrg = UsersMap.ToOriginalID(user_id);
			var userPosFeedback = _trainUserPosFeedback[userIdOrg];

			var sampleFeedback = userPosFeedback[random.Next(userPosFeedback.Count)];
			item_id = ItemsMap.ToInternalID(sampleFeedback.Item.Id);

			var allUserFeedback = userPosFeedback.Concat(_trainUserNegFeedback[userIdOrg]).ToList();

			var toSampleLevels = allUserFeedback.Select(f => f.Level).Distinct().Where(l => l > sampleFeedback.Level).ToList();
			int sampleLevel = random.Next(toSampleLevels.Count + 1);

			// if there is no lower level feedbacks, sample from unobserved
			if (sampleLevel == toSampleLevels.Count)
			{
				SampleOtherItem(user_id, item_id, out other_item_id);
				_numUnobservedSamples++;
			}
			else
			{
				var toSample = allUserFeedback.Where(f => f.Level == toSampleLevels[sampleLevel]).ToList();
				other_item_id = ItemsMap.ToInternalID(toSample[random.Next(toSample.Count)].Item.Id);
				_numObservedSamples++;
			}
		}

		public override void Train()
		{
			Logger.Current.Info("Training with MultiLevel BPR...");
			base.Train();
			Logger.Current.Info("Num unobserved negative samples: {0}, observed negative samples: {1}", 
				_numUnobservedSamples, _numObservedSamples);
		}

	}
}
