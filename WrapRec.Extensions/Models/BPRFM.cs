using MyMediaLite.ItemRecommendation;
using MyMediaLite.DataType;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite;
using WrapRec.Data;
using MyMediaLite.Data;
using WrapRec.Models;
using WrapRec.Core;

namespace WrapRec.Extensions.Models
{
	public class BPRFM : BPRMF
	{
		public Split Split { get; set; }
        public Model Model { get; set; }
        public Mapping UsersMap { get; set; }
		public Mapping ItemsMap { get; set; }
		public FmFeatureBuilder FeatureBuilder { get; set; }
		public int NumTrainFeaturs { get; protected set; }

		// regularization that is considered for auxiliary features
		public float RegC { get { return reg_c; } set { reg_c = value; } }
		protected float reg_c = 0.00025f;

		protected Matrix<float> feature_factors;

        // this method will be called by MML train method thus the data is already loaded and features are translated
		protected override void InitModel()
		{
			base.InitModel();
			NumTrainFeaturs = FeatureBuilder.Mapper.NumberOfEntities;
			feature_factors = new Matrix<float>(NumTrainFeaturs, NumFactors);
			feature_factors.InitNormal(InitMean, InitStdDev);
		}

		protected override void UpdateFactors(int user_id, int item_id, int other_item_id, bool update_u, bool update_i, bool update_j)
		{
			// used by WrapRec-based logic
			string userIdOrg = UsersMap.ToOriginalID(user_id);
			string itemIdOrg = ItemsMap.ToOriginalID(item_id);

		    List<Tuple<int, float>> features = new List<Tuple<int, float>>();
            if (Split.SetupParameters.ContainsKey("feedbackAttributes"))
                features = Split.Container.FeedbacksDic[userIdOrg, itemIdOrg].GetAllAttributes().Select(a => a.Translation).ToList();

			double item_bias_diff = item_bias[item_id] - item_bias[other_item_id];

			double y_uij = item_bias_diff + MatrixExtensions.RowScalarProductWithRowDifference(
				user_factors, user_id, item_factors, item_id, item_factors, other_item_id);

            foreach (var feat in features)
            {
                y_uij += feat.Item2 * MatrixExtensions.RowScalarProductWithRowDifference(
                    feature_factors, feat.Item1, item_factors, item_id, item_factors, other_item_id);
            }

			double exp = Math.Exp(y_uij);
			double sigmoid = 1 / (1 + exp);

			// adjust bias terms
			if (update_i)
			{
				// TODO: check why -Bias
				double update = sigmoid - BiasReg * item_bias[item_id];
				item_bias[item_id] += (float)(learn_rate * update);
			}

			if (update_j)
			{
				double update = -sigmoid - BiasReg * item_bias[other_item_id];
				item_bias[other_item_id] += (float)(learn_rate * update);
			}

			// adjust factors
			for (int f = 0; f < num_factors; f++)
			{
				float v_uf = user_factors[user_id, f];
				float v_if = item_factors[item_id, f];
				float v_jf = item_factors[other_item_id, f];

				if (update_u)
				{
					double update = (v_if - v_jf) * sigmoid - reg_u * v_uf;
					user_factors[user_id, f] = (float)(v_uf + learn_rate * update);
				}

				// update features latent factors and make a sum term to use later for updating item factors
				// sum = Sum_{l=1}{num_features} c_l * v_{c_l,f}
				float sum = 0f;

				foreach (var feat in features)
				{
					float v_zf = feature_factors[feat.Item1, f];
					float x_z = feat.Item2;

					sum += x_z * v_zf;

					double update = x_z * (v_if - v_jf) * sigmoid - reg_c * v_zf;
					feature_factors[feat.Item1, f] = (float)(v_zf + learn_rate * update);
				}

				if (update_i)
				{
					double update = (v_uf + sum) * sigmoid - reg_i * v_if;
					item_factors[item_id, f] = (float)(v_if + learn_rate * update);
				}

				if (update_j)
				{
					double update = (-v_uf - sum) * sigmoid - reg_j * v_jf;
					item_factors[other_item_id, f] = (float)(v_jf + learn_rate * update);
				}
			}
		}

        public override void Iterate()
        {
            int time = (int) Wrap.MeasureTime(delegate() { base.Iterate(); }).TotalMilliseconds;
            Model.OnIterate(this, time);
        }

        public override float Predict(int user_id, int item_id)
		{
			bool newUser = (user_id > MaxUserID);
			bool newItem = (item_id > MaxItemID);

			float itemBias = newItem ? 0 : item_bias[item_id];
			float userItemTerm = (newUser || newItem) ? 0 : MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

			return itemBias + userItemTerm;
		}

		public float Predict(Feedback feedback)
		{
			int userId = UsersMap.ToInternalID(feedback.User.Id);
			int itemId = ItemsMap.ToInternalID(feedback.Item.Id);
			var featurs = feedback.GetAllAttributes().Select(a => FeatureBuilder.TranslateAttribute(a));

			bool newUser = (userId > MaxUserID);
			bool newItem = (itemId > MaxItemID);

			float userAttrsTerm = 0, itemAttrsTerm = 0;

			foreach (var feat in featurs)
			{
				// if feat_index is greater than MaxFeatureId it means that the feature is new in test set so its factors has not been learnt
				if (feat.Item1 < NumTrainFeaturs)
				{
					float x_z = feat.Item2;
					
					userAttrsTerm += newUser ? 0 : x_z * MatrixExtensions.RowScalarProduct(feature_factors, feat.Item1, user_factors, userId);
					itemAttrsTerm += newItem ? 0 : x_z * MatrixExtensions.RowScalarProduct(feature_factors, feat.Item1, item_factors, itemId);
				}
			}

			return Predict(userId, itemId) + userAttrsTerm + itemAttrsTerm;
		}
	}
}
