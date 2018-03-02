using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite.RatingPrediction;
using WrapRec.Core;
using WrapRec.Data;
using WrapRec.Models;

namespace WrapRec.Extensions.Models
{
    public class MmlFmRecommender : MmlRecommender
    {

        public override void Setup()
        {
            base.Setup();

            if (!(MmlRecommenderInstance is FM))
                throw new WrapRecException("Expect ml-class 'FM' for 'MmlFmRecommender'");

            var wFm = MmlRecommenderInstance as WFM;

            if (wFm != null)
                wFm.NumGroups = int.Parse(SetupParameters["numGroups"]);

            if (SetupParameters.ContainsKey("Normalize"))
                ((FM)MmlRecommenderInstance).Normalize = bool.Parse(SetupParameters["Normalize"]);
        }

        public override void Train(Split split)
        {
            var mmlInstance = (FM)MmlRecommenderInstance;
            var featBuilder = new FmFeatureBuilder();

            var wFm = MmlRecommenderInstance as WeightedBPRFM;

            if (DataType == WrapRec.IO.DataType.Ratings)
            {
                var mmlFeedback = new Ratings();
                foreach (var feedback in split.Train)
                {
                    var rating = (Rating)feedback;
                    mmlFeedback.Add(UsersMap.ToInternalID(rating.User.Id), ItemsMap.ToInternalID(rating.Item.Id), rating.Value);

                    // the attributes are translated so that they can be used later for training
                    foreach (var attr in feedback.GetAllAttributes())
                    {
                        attr.Translation = featBuilder.TranslateAttribute(attr);
                        // hard code attribute group. User is 0, item is 1, others is 2
                        attr.Group = 2;
                        if (wFm != null && !wFm.FeatureGroups.ContainsKey(attr.Translation.Item1))
                            wFm.FeatureGroups.Add(attr.Translation.Item1, 2);
                    }
                }
                ((IRatingPredictor)MmlRecommenderInstance).Ratings = mmlFeedback;
            }

            foreach (var feedback in split.Test)
            {
                // the attributes are translated so that they can be used later for training
                foreach (var attr in feedback.GetAllAttributes())
                {
                    attr.Translation = featBuilder.TranslateAttribute(attr);
                    // hard code attribute group. User is 0, item is 1, others is 2
                    attr.Group = 2;
                    if (wFm != null && !wFm.FeatureGroups.ContainsKey(attr.Translation.Item1))
                        wFm.FeatureGroups.Add(attr.Translation.Item1, 2);

                }
            }

            mmlInstance.Split = split;
            mmlInstance.Model = this;
            mmlInstance.UsersMap = UsersMap;
            mmlInstance.ItemsMap = ItemsMap;
            mmlInstance.FeatureBuilder = featBuilder;

            Logger.Current.Trace("Training with MmlFmRecommender recommender...");
            PureTrainTime = (int)Wrap.MeasureTime(delegate () { mmlInstance.Train(); }).TotalMilliseconds;
        }

    }
}
