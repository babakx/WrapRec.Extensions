using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite;
using MyMediaLite.DataType;
using WrapRec.Utils;

namespace WrapRec.Extensions.Models
{
    public class WFM : FM
    {
        protected float[] weights;
        public float reg_w { get; set; }

        public Dictionary<int, int> FeatureGroups;
        //private StreamWriter _alphaWriter;
        private int _iter = 0;
        public int NumGroups { get; set; }

        public WFM()
        {
            FeatureGroups = new Dictionary<int, int>();
            Regularization = 0.0025f;
            reg_w = 0.0025f;
        }

        protected override void InitModel()
        {
            base.InitModel();
            //_alphaWriter = new StreamWriter("alpha_" + Split.Container.Id + ".csv");
            weights = new float[NumGroups];
            var r = new System.Random();
            // initialized weights
            for (int i = 0; i < NumGroups; i++)
            {
                weights[i] = (float)r.NextDouble();
            }
            NormalizeWeights();
        }

        private void NormalizeWeights()
        {
            for (int i = 0; i < weights.Length; i++)
                if (weights[i] < 0)
                    weights[i] = 0;

            float sum = weights.Sum() / NumGroups;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = weights[i] / sum;
            }
        }

        public override void Iterate()
        {
            int time = (int)Wrap.MeasureTime(delegate () { base.Iterate(); }).TotalMilliseconds;
            Model.OnIterate(this, time);
            //_alphaWriter.Flush();
        }

        protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
        {
            int time = (int)Wrap.MeasureTime(delegate () { _Iterate(rating_indices, update_user, update_item); }).TotalMilliseconds;
            Model.OnIterate(this, time);
        }


        protected void _Iterate(IList<int> rating_indices, bool update_user, bool update_item)
        {
            foreach (int index in rating_indices)
            {
                int u = ratings.Users[index];
                int i = ratings.Items[index];

                int g_u = 0; //FeatureGroups[user_id];
                int g_i = 1; //FeatureGroups[item_id];
                float alpha_u = weights[g_u];
                float alpha_i = weights[g_i];

                // used by WrapRec-based logic
                string userIdOrg = UsersMap.ToOriginalID(u);
                string itemIdOrg = ItemsMap.ToOriginalID(i);

                List<Tuple<int, float>> features = new List<Tuple<int, float>>();
                if (Split.SetupParameters.ContainsKey("feedbackAttributes"))
                    features = Split.Container.FeedbacksDic[userIdOrg, itemIdOrg].GetAllAttributes().Select(a => a.Translation)
                        .NormalizeSumToOne(Normalize).ToList();

                var p = Predict(u, i);

                float err = (p - ratings[index]) * 2;

                float sum_u = 0, sum_i = 0;
                foreach (var feature in features)
                {
                    int j = feature.Item1;
                    float x_j = feature.Item2;
                    int g_z = 2; // FeatureGroups[feature.Item1];
                    float alpha_z = weights[g_z];

                    sum_u += x_j*alpha_z*MatrixExtensions.RowScalarProduct(user_factors, u, feature_factors, j);
                    sum_i += x_j*alpha_z*MatrixExtensions.RowScalarProduct(item_factors, i, feature_factors, j);
                }
                float ui = MatrixExtensions.RowScalarProduct(item_factors, i, user_factors, u);
                sum_u += alpha_i * ui;
                sum_i += alpha_u * ui;

                float sum_z = 0;
                float sum_z_bias = 0;
                for (int z = 0; z < features.Count; z++)
                {
                    int z_ix = features[z].Item1;
                    float x_z = features[z].Item2;
                    float sum_j = 0;
                    int g_z = 2; // FeatureGroups[z_ix];
                    float alpha_z = weights[g_z];
                    for (int j = z + 1; j < features.Count; j++)
                    {
                        int j_ix = features[j].Item1;
                        float x_j = features[j].Item2;
                        sum_j += x_j*MatrixExtensions.RowScalarProduct(feature_factors, z_ix, feature_factors, j_ix);
                    }
                    sum_z += 2*alpha_z*x_z*sum_j;
                    sum_z += x_z*alpha_u*MatrixExtensions.RowScalarProduct(feature_factors, z_ix, user_factors, u);
                    sum_z += x_z*alpha_i*MatrixExtensions.RowScalarProduct(feature_factors, z_ix, item_factors, i);
                    sum_z_bias += x_z*feature_biases[z_ix];
                }


                float[] sum = new float[NumFactors];
                foreach (var feature in features)
                {
                    int j = feature.Item1;
                    float x_j = feature.Item2;
                    int g_z = 2; //FeatureGroups[feature.Item1];
                    float alpha_z = weights[g_z];

                    for (int f = 0; f < NumFactors; f++)
                        sum[f] += feature_factors[j, f] * x_j * alpha_z;
                }

                for (int f = 0; f < NumFactors; f++)
                    sum[f] += user_factors[u, f] * alpha_u + item_factors[i, f] * alpha_i;

                // adjust biases
                global_bias -= current_learnrate*(err + RegB*global_bias);

                if (update_user)
                    user_bias[u] -= current_learnrate * (err * alpha_u + RegU * user_bias[u]);
                if (update_item)
                    item_bias[i] -= current_learnrate * (err * alpha_i + RegI * item_bias[i]);

                foreach (var feature in features)
                {
                    int j = feature.Item1;
                    float x_j = feature.Item2;
                    float w_j = feature_biases[j];
                    int g_z = 2; // FeatureGroups[feature.Item1];
                    float alpha_z = weights[g_z];

                    feature_biases[j] -= current_learnrate * (x_j * alpha_z * err + RegC * w_j);
                }

                // adjust latent factors
                for (int f = 0; f < NumFactors; f++)
                {
                    double v_uf = user_factors[u, f];
                    double v_if = item_factors[i, f];

                    if (update_user)
                    {
                        double delta_u = alpha_u * (sum[f] - v_uf * alpha_u) * err + RegU * v_uf;
                        user_factors.Inc(u, f, -current_learnrate * delta_u);
                    }
                    if (update_item)
                    {
                        double delta_i = alpha_i * (sum[f] - v_if * alpha_i) * err + RegI * v_if;
                        item_factors.Inc(i, f, -current_learnrate * delta_i);
                    }

                    foreach (var feature in features)
                    {
                        int j = feature.Item1;
                        float x_j = feature.Item2;
                        float v_jf = feature_factors[j, f];
                        int g_z = 2; // FeatureGroups[feature.Item1];
                        float alpha_z = weights[g_z];

                        double delta_j = x_j * alpha_z * (sum[f] - v_jf * x_j * alpha_z) * err + RegC * v_jf;
                        feature_factors.Inc(j, f, -current_learnrate * delta_j);
                    }
                }

                // update alphas
                float update_alpha_u = (user_bias[u] + sum_u)*err + reg_w * weights[g_u];
                weights[g_u] -= current_learnrate*update_alpha_u;

                float update_alpha_i = (item_bias[i] + sum_i)*err + reg_w * weights[g_i];
                weights[g_i] -= current_learnrate * update_alpha_i;

                for (int g = 0; g < NumGroups - 2; g++)
                {
                    float alpha_z_g = weights[g + 2];
                    float update_alpha_z = (sum_z + sum_z_bias)*err + reg_w*alpha_z_g;
                    weights[g + 2] -= current_learnrate*update_alpha_z;
                }


                NormalizeWeights();
            }

            Console.WriteLine($"alpha_u: {weights[0]}, alpha_i: {weights[1]}" + (weights.Length > 2 ? $", alpha_z: {weights[2]}" : ""));
            //_alphaWriter.WriteLine(++_iter + "," + weights[0] + "," + weights[1] + (weights.Length > 2 ? "," + weights[2] : ""));
        }

        public override float Predict(int user_id, int item_id)
        {
            int u = user_id;
            int i = item_id;

            // used by WrapRec-based logic
            string userIdOrg = UsersMap.ToOriginalID(user_id);
            string itemIdOrg = ItemsMap.ToOriginalID(item_id);

            List<Tuple<int, float>> features = new List<Tuple<int, float>>();
            if (Split.SetupParameters.ContainsKey("feedbackAttributes"))
                features = Split.Container.FeedbacksDic[userIdOrg, itemIdOrg].GetAllAttributes().Select(a => a.Translation)
                    .NormalizeSumToOne(Normalize).ToList();

            float alpha_u = weights[0];
            float alpha_i = weights[1];
            float alpha_z = weights.Length > 2 ? weights[2] : 0;

            float score = global_bias;

            if (u < user_bias.Length)
                score += user_bias[u] * alpha_u;

            if (i < item_bias.Length)
                score += item_bias[i] * alpha_i;

            foreach (var feat in features)
            {
                float x_z = feat.Item2;
                score += feature_biases[feat.Item1] * x_z * alpha_z;
            }

            for (int f = 0; f < NumFactors; f++)
            {
                float sum = 0, sum_sq = 0;

                float v_uf = 0, v_if = 0;
                if (u < user_bias.Length)
                    v_uf = user_factors[u, f];
                if (i < item_bias.Length)
                    v_if = item_factors[i, f];

                sum += v_uf * alpha_u + v_if * alpha_i;
                sum_sq += v_uf * v_uf * alpha_u * alpha_u + v_if * v_if * alpha_i * alpha_i;

                foreach (var feat in features)
                {
                    int j = feat.Item1;
                    float x_j = feat.Item2;
                    float v_jf = feature_factors[j, f];

                    sum += x_j * v_jf * alpha_z;
                    sum_sq += x_j * x_j * v_jf * v_jf * alpha_z * alpha_z;
                }

                score += 0.5f * (sum * sum - sum_sq);
            }

            if (score > MaxRating)
                return MaxRating;
            if (score < MinRating)
                return MinRating;

            return score;
        }

    }
}
