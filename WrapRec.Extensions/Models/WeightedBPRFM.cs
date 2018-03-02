using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite.DataType;
using MyMediaLite.ItemRecommendation;
using WrapRec.Core;
using WrapRec.Utils;

namespace WrapRec.Extensions.Models
{
    public class WeightedBPRFM : BPRFM
    {
        protected float[] weights;
        public float reg_w { get; set; } 

        public Dictionary<int, int> FeatureGroups;
        //private StreamWriter _alphaWriter;
        private int _iter = 0;

        public int NumGroups { get; set; }

        public WeightedBPRFM()
        {
            FeatureGroups = new Dictionary<int, int>();
            reg_w = 0.00025f;
        }

        protected override void InitModel()
        {
            base.InitModel();
            weights = new float[NumGroups];
            //_alphaWriter = new StreamWriter("alpha_" + Split.Container.Id + ".csv");

            var r = new System.Random();
            // initialized weights
            for (int i = 0; i < NumGroups; i++)
            {
                weights[i] = (float)r.NextDouble();
            }
            NormalizeWeights();
        }

        protected override void UpdateFactors(int user_id, int item_id, int other_item_id, bool update_u, bool update_i, bool update_j)
        {
            // used by WrapRec-based logic
            string userIdOrg = UsersMap.ToOriginalID(user_id);
            string itemIdOrg = ItemsMap.ToOriginalID(item_id);

            List<Tuple<int, float>> features = new List<Tuple<int, float>>();
            if (Split.SetupParameters.ContainsKey("feedbackAttributes"))
                features = Split.Container.FeedbacksDic[userIdOrg, itemIdOrg].GetAllAttributes().Select(a => a.Translation).NormalizeSumToOne(Normalize).ToList();

            double item_bias_diff = item_bias[item_id] - item_bias[other_item_id];

            int g_u = 0; //FeatureGroups[user_id];
            int g_i = 1; //FeatureGroups[item_id];
            float alpha_u = weights[g_u];
            float alpha_i = weights[g_i];

            double u_i_term = MatrixExtensions.RowScalarProductWithRowDifference(
                user_factors, user_id, item_factors, item_id, item_factors, other_item_id);

            double y_uij = item_bias_diff + alpha_u*alpha_i*u_i_term;

            double items_z_term_sum = 0;
            double[] items_z_terms = new double[features.Count];
            double[] group_z_terms = new double[NumGroups - 2];
            int z = 0;
            foreach (var feat in features)
            {
                int g_z = FeatureGroups[feat.Item1];
                float alpha_z = weights[g_z];
                items_z_terms[z] = feat.Item2 * MatrixExtensions.RowScalarProductWithRowDifference(
                    feature_factors, feat.Item1, item_factors, item_id, item_factors, other_item_id);
                group_z_terms[g_z - 2] += items_z_terms[z];
                items_z_term_sum += alpha_z*items_z_terms[z];
                z++;
            }
            y_uij += alpha_i * items_z_term_sum;

            double exp = Math.Exp(y_uij);
            double sigmoid = 1 / (1 + exp);

            // adjust bias terms
            if (update_i)
            {
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
                    double update = alpha_u * alpha_i * (v_if - v_jf) * sigmoid - reg_u * v_uf;
                    user_factors[user_id, f] = (float)(v_uf + learn_rate * update);
                }

                // update features latent factors and make a sum term to use later for updating item factors
                // sum = Sum_{l=1}{num_features} c_l * v_{c_l,f}
                float sum = 0f;

                foreach (var feat in features)
                {
                    float v_zf = feature_factors[feat.Item1, f];
                    float x_z = feat.Item2;
                    int g_z = FeatureGroups[feat.Item1];
                    float alpha_z = weights[g_z];

                    sum += x_z * v_zf * alpha_z;

                    double update = alpha_i * alpha_z * x_z * (v_if - v_jf) * sigmoid - reg_c * v_zf;
                    feature_factors[feat.Item1, f] = (float)(v_zf + learn_rate * update);
                }

                if (update_i)
                {
                    double update = (alpha_u * alpha_i * v_uf + alpha_i * sum) * sigmoid - reg_i * v_if;
                    item_factors[item_id, f] = (float)(v_if + learn_rate * update);
                }

                if (update_j)
                {
                    double update = (alpha_u * alpha_i * -v_uf - alpha_i * sum) * sigmoid - reg_j * v_jf;
                    item_factors[other_item_id, f] = (float)(v_jf + learn_rate * update);
                }
            }

            // update alphas
            double update_alpha_u = alpha_i * u_i_term * sigmoid - reg_w * alpha_u;
            weights[g_u] = (float)(alpha_u + learn_rate * update_alpha_u);

            //NormalizeWeights();

            double update_alpha_i = (alpha_u * u_i_term + items_z_term_sum) * sigmoid - reg_w * alpha_i;
            weights[g_i] = (float)(alpha_i + learn_rate * update_alpha_i);

            for (int g = 0; g < NumGroups - 2; g++)
            {
                double alpha_z_g = weights[g + 2];
                double update_alpha_z_g = alpha_i*group_z_terms[g]*sigmoid - reg_w*alpha_z_g;
                weights[g + 2] = (float) (alpha_z_g + learn_rate*update_alpha_z_g);
            }

            // normalize weights
            NormalizeWeights();

        }

        private void NormalizeWeights()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] < 0)
                    weights[i] = 0;
            }

            float sum = weights.Sum() / NumGroups;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = weights[i]/sum;
            }
        }

        public override void Iterate()
        {
            Console.WriteLine($"alpha_u: {weights[0]}, alpha_i: {weights[1]}" + (weights.Length > 2 ? $", alpha_z: {weights[2]}" : ""));
            base.Iterate();
            //_alphaWriter.WriteLine(++_iter + "," + weights[0] + "," + weights[1] + (weights.Length > 2 ? "," + weights[2] : ""));
            //_alphaWriter.Flush();
        }

        public override float Predict(int user_id, int item_id)
        {
            string userIdOrg = UsersMap.ToOriginalID(user_id);
            string itemIdOrg = ItemsMap.ToOriginalID(item_id);
            List<Tuple<int, float>> features = new List<Tuple<int, float>>();

            if (Split.Container.FeedbacksDic.ContainsKey(userIdOrg, itemIdOrg))
            {
                var feedback = Split.Container.FeedbacksDic[userIdOrg, itemIdOrg];
                features = feedback.GetAllAttributes().Select(a => FeatureBuilder.TranslateAttribute(a)).NormalizeSumToOne(Normalize).ToList();
            }

            bool newUser = (user_id > MaxUserID);
            bool newItem = (item_id > MaxItemID);

            float userAttrsTerm = 0, itemAttrsTerm = 0;

            foreach (var feat in features)
            {
                // if feat_index is greater than MaxFeatureId it means that the feature is new in test set so its factors has not been learnt
                if (feat.Item1 < NumTrainFeaturs)
                {
                    int g_z = FeatureGroups[feat.Item1];
                    float alpha_z = weights[g_z];
                    float x_z = feat.Item2;

                    userAttrsTerm += newUser ? 0 : alpha_z * x_z * MatrixExtensions.RowScalarProduct(feature_factors, feat.Item1, user_factors, user_id);
                    itemAttrsTerm += newItem ? 0 : alpha_z * x_z * MatrixExtensions.RowScalarProduct(feature_factors, feat.Item1, item_factors, item_id);
                }
            }

            int g_u = 0; //FeatureGroups[user_id];
            int g_i = 1; //FeatureGroups[item_id];
            float alpha_u = weights[g_u];
            float alpha_i = weights[g_i];

            float itemBias = newItem ? 0 : item_bias[item_id];
            float userItemTerm = (newUser || newItem) ? 0 : alpha_u * alpha_i * MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

            return itemBias + userItemTerm + alpha_u * userAttrsTerm + alpha_i * itemAttrsTerm;
        }

        public override float Predict(Feedback feedback)
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
                    int g_z = FeatureGroups[feat.Item1];
                    float alpha_z = weights[g_z];

                    userAttrsTerm += newUser ? 0 : alpha_z * x_z * MatrixExtensions.RowScalarProduct(feature_factors, feat.Item1, user_factors, userId);
                    itemAttrsTerm += newItem ? 0 : alpha_z * x_z * MatrixExtensions.RowScalarProduct(feature_factors, feat.Item1, item_factors, itemId);
                }
            }

            int u = 0;
            int i = 1;
            float alpha_u = weights[u];
            float alpha_i = weights[i];

            float itemBias = newItem ? 0 : item_bias[itemId];
            float userItemTerm = (newUser || newItem) ? 0 : alpha_u * alpha_i * MatrixExtensions.RowScalarProduct(user_factors, userId, item_factors, itemId);

            return itemBias + userItemTerm + alpha_u * userAttrsTerm + alpha_i * itemAttrsTerm;
        }

    }
}
