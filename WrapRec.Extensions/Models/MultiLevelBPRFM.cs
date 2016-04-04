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

		protected override void InitModel()
		{
			base.InitModel();
			_feedbackLevels = Split.Train.Select(f => f.Level).Distinct().ToArray();
		}

		protected override void SampleTriple(out int user_id, out int item_id, out int other_item_id)
		{
			user_id = SampleUser();
			string userIdOrg = UsersMap.ToOriginalID(user_id);
			var userFeedback = Split.Container.Users[userIdOrg].Feedbacks.Where(f => f.SliceType == FeedbackSlice.TRAIN).ToList();

			var sampleFeedback = userFeedback[random.Next(userFeedback.Count)];
			item_id = ItemsMap.ToInternalID(sampleFeedback.Item.Id);

			var toSampleLevels = userFeedback.Select(f => f.Level).Distinct().Where(l => l > sampleFeedback.Level).ToList();
			int sampleLevel = random.Next(toSampleLevels.Count + 1);

			// if there is no lower level feedbacks, sample from unobserved
			if (sampleLevel == toSampleLevels.Count)
			{
				SampleOtherItem(user_id, item_id, out other_item_id);
				_numUnobservedSamples++;
			}
			else
			{
				var toSample = userFeedback.Where(f => f.Level == toSampleLevels[sampleLevel]).ToList();
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
