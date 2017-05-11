using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WrapRec.Core;
using WrapRec.Data;
using WrapRec.Models;

namespace WrapRec.Extensions.Models
{
	public class MmlBprfmRecommender : MmlRecommender
	{
		public MmlBprfmRecommender()
			: base()
		{ }

		public override void Setup()
		{
			base.Setup();

			if (!(MmlRecommenderInstance is BPRFM))
				throw new WrapRecException("Expect ml-class 'BPRFM' for 'MmlBprfmRecommender'");

            var wBprFm = MmlRecommenderInstance as WeightedBPRFM;

		    if (wBprFm != null)
		        wBprFm.NumGroups = int.Parse(SetupParameters["numGroups"]);
		}

        public override void Train(Split split)
		{
			var mmlInstance = (BPRFM)MmlRecommenderInstance;
			var featBuilder = new FmFeatureBuilder();
			var mmlFeedback = new PosOnlyFeedback<SparseBooleanMatrix>();

		    var wBprFm = MmlRecommenderInstance as WeightedBPRFM;

			foreach (var feedback in split.Train)
			{
				mmlFeedback.Add(UsersMap.ToInternalID(feedback.User.Id), ItemsMap.ToInternalID(feedback.Item.Id));
				
				// the attributes are translated so that they can be used later for training
			    foreach (var attr in feedback.GetAllAttributes())
			    {
			        attr.Translation = featBuilder.TranslateAttribute(attr);
                    // hard code attribute group. User is 0, item is 1, others is 2
                    attr.Group = 2;
                    if (wBprFm != null && !wBprFm.FeatureGroups.ContainsKey(attr.Translation.Item1))
			            wBprFm.FeatureGroups.Add(attr.Translation.Item1, 2);
			    }
			}

            foreach (var feedback in split.Test)
            {
                // the attributes are translated so that they can be used later for training
                foreach (var attr in feedback.GetAllAttributes())
                {
                    attr.Translation = featBuilder.TranslateAttribute(attr);
                    // hard code attribute group. User is 0, item is 1, others is 2
                    attr.Group = 2;
                    if (wBprFm != null && !wBprFm.FeatureGroups.ContainsKey(attr.Translation.Item1))
                        wBprFm?.FeatureGroups.Add(attr.Translation.Item1, 2);
                }
            }

            mmlInstance.Feedback = mmlFeedback;
			mmlInstance.Split = split;
            mmlInstance.Model = this;
			mmlInstance.UsersMap = UsersMap;
			mmlInstance.ItemsMap = ItemsMap;
			mmlInstance.FeatureBuilder = featBuilder;

			Logger.Current.Trace("Training with MmlBprfmRecommender recommender...");
			PureTrainTime = (int)Wrap.MeasureTime(delegate() { mmlInstance.Train(); }).TotalMilliseconds;
		}

	    public override float Predict(Feedback feedback)
		{
			return ((BPRFM)MmlRecommenderInstance).Predict(feedback);
		}

	}
}
