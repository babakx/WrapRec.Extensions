using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.RatingPrediction;
using WrapRec.Data;
using WrapRec.Models;
using MyMediaLite.DataType;
using WrapRec.Utils;
using MyMediaLite;

namespace WrapRec.Extensions.Models
{
    public class FM : BiasedMatrixFactorization
    {
        public Split Split { get; set; }
        public Model Model { get; set; }
        public Mapping UsersMap { get; set; }
        public Mapping ItemsMap { get; set; }
        public FmFeatureBuilder FeatureBuilder { get; set; }
        public int NumTrainFeaturs { get; protected set; }

        // regularization that is considered for auxiliary features
        public float RegC { get { return reg_c; } set { reg_c = value; } }
        protected float reg_c = 0.0025f;
        public float RegB { get { return reg_b; } set { reg_b = value; } }
        protected float reg_b = 0.0025f;

        protected Matrix<float> feature_factors;

        protected float[] feature_biases;
        public bool Normalize { get; set; }


        protected override void InitModel()
        {
            base.InitModel();
            NumTrainFeaturs = FeatureBuilder.Mapper.NumberOfEntities;
            feature_factors = new Matrix<float>(NumTrainFeaturs, NumFactors);
            feature_factors.InitNormal(InitMean, InitStdDev);
            feature_biases = new float[NumTrainFeaturs];
            RegU = 0.0015f;
            RegI = 0.0015f;
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

                // used by WrapRec-based logic
                string userIdOrg = UsersMap.ToOriginalID(u);
                string itemIdOrg = ItemsMap.ToOriginalID(i);

                List<Tuple<int, float>> features = new List<Tuple<int, float>>();
                if (Split.SetupParameters.ContainsKey("feedbackAttributes"))
                    features = Split.Container.FeedbacksDic[userIdOrg, itemIdOrg].GetAllAttributes()
                        .Select(a => a.Translation).NormalizeSumToOne(Normalize).ToList();

                var p = Predict(u, i);
                float err = (p - ratings[index])*2;

                float[] sum = new float[NumFactors];
                foreach (var feature in features)
                {
                    int j = feature.Item1;
                    float x_j = feature.Item2;

                    for (int f = 0; f < NumFactors; f++)
                        sum[f] += feature_factors[j, f] * x_j;
                }

                for (int f = 0; f < NumFactors; f++)
                    sum[f] += user_factors[u, f] + item_factors[i, f];

                // adjust biases
                //global_bias -= current_learnrate*(err + RegB*global_bias);

                if (update_user)
                    user_bias[u] -= current_learnrate*(err + RegU*user_bias[u]);
                if (update_item)
                    item_bias[i] -= current_learnrate*(err + RegI*item_bias[i]);

                foreach (var feature in features)
                {
                    int j = feature.Item1;
                    float x_j = feature.Item2;
                    float w_j = feature_biases[j];

                    feature_biases[j] -= BiasLearnRate*current_learnrate*(x_j * err + BiasReg*RegC*w_j);
                }

                // adjust latent factors
                for (int f = 0; f < NumFactors; f++)
                {
                    double v_uf = user_factors[u, f];
                    double v_if = item_factors[i, f];

                    if (update_user)
                    {
                        double delta_u = (sum[f] - v_uf)*err + RegU * v_uf;
                        user_factors.Inc(u, f, -current_learnrate * delta_u);
                    }
                    if (update_item)
                    {
                        double delta_i = (sum[f] - v_if)*err + RegI * v_if;
                        item_factors.Inc(i, f, -current_learnrate * delta_i);
                    }

                    foreach (var feature in features)
                    {
                        int j = feature.Item1;
                        float x_j = feature.Item2;
                        float v_jf = feature_factors[j, f];

                        double delta_j = (sum[f]*x_j - v_jf*x_j*x_j)*err + RegC*v_jf;
                        feature_factors.Inc(j, f, -current_learnrate*delta_j);
                    }
                }
            }
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
                features = Split.Container.FeedbacksDic[userIdOrg, itemIdOrg].GetAllAttributes()
                    .Select(a => a.Translation).NormalizeSumToOne(Normalize).ToList();

            float score = global_bias;

            if (u < user_bias.Length)
                score += user_bias[u];

            if (i < item_bias.Length)
                score += item_bias[i];

            foreach (var feat in features)
            {
                score += feature_biases[feat.Item1];
            }

            for (int f = 0; f < NumFactors; f++)
            {
                float sum = 0, sum_sqrt = 0;

                float v_uf = 0, v_if = 0;
                if (u < user_bias.Length)
                    v_uf = user_factors[u, f];
                if (i < item_bias.Length)
                    v_if = item_factors[i, f];

                sum += v_uf + v_if;
                sum_sqrt += v_uf*v_uf + v_if*v_if;

                foreach (var feat in features)
                {
                    int j = feat.Item1;
                    float x_j = feat.Item2;
                    float v_jf = feature_factors[j, f];

                    sum += x_j*v_jf;
                    sum_sqrt += x_j*x_j*v_jf*v_jf;
                }

                score += 0.5f*(sum*sum - sum_sqrt);
            }

            if (score > MaxRating)
                return MaxRating;
            if (score < MinRating)
                return MinRating;

            return score;
        }
    }
}
