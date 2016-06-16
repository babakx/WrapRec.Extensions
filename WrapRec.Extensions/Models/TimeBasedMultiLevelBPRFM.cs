using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WrapRec.Core;

namespace WrapRec.Extensions.Models
{
	public class TimeBasedMultiLevelBPRFM : MultiLevelBPRFM
	{
		public Dictionary<string, List<Feedback>> UserTimeSortedFeedback { get; private set; }

		Dictionary<Feedback, int> _indexInSortedFeedback;

		protected override void InitModel()
		{
			base.InitModel();
			
			UserTimeSortedFeedback = UserFeedback.ToDictionary(kv => kv.Key,
				kv => kv.Value.OrderByDescending(f => f.Attributes["timestamp"].Value).ToList());

			_indexInSortedFeedback = new Dictionary<Feedback, int>();

			foreach (string u in UserTimeSortedFeedback.Keys)
			{
				for (int i = 0; i < UserTimeSortedFeedback[u].Count; i++)
				{
					_indexInSortedFeedback.Add(UserTimeSortedFeedback[u][i], i);
				}
			}	
		}
		
		protected override Feedback SampleNegFeedback(Feedback posFeedback)
		{
			int observedOrUnobserved = _unobservedOrNegativeSampler.Sample();
			if (observedOrUnobserved == 1)
				return SampleUnobservedNegFeedback(posFeedback);

			string userId = posFeedback.User.Id;
			int index = _indexInSortedFeedback[posFeedback];
			int count = UserTimeSortedFeedback[userId].Count;

			if (index >= count - 1)
				return SampleUnobservedNegFeedback(posFeedback);

			int negIndex = random.Next(count - index - 1) + 1;
			return UserTimeSortedFeedback[userId][negIndex];
		}
	}
}
