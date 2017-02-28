using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite;
using MyMediaLite.ItemRecommendation;
using WrapRec.Models;
using MyMediaLite.DataType;

namespace WrapRec.Extensions.Models
{
    public class WrapRecBPRMF : BPRMF, IModelAwareRecommender
    {
        public Model Model { get; set; }

        public override void Iterate()
        {
            int time = (int)Wrap.MeasureTime(delegate () { base.Iterate(); }).TotalMilliseconds;
            Model.OnIterate(this, time);
        }

        protected override void UpdateFactors(int user_id, int item_id, int other_item_id, bool update_u, bool update_i, bool update_j)
        {
            double x_uij = MyMediaLite.DataType.MatrixExtensions.RowScalarProductWithRowDifference(user_factors, user_id, item_factors, item_id, item_factors, other_item_id);
            double one_over_one_plus_ex = 1 / (1 + Math.Exp(x_uij));

            // adjust factors
            for (int f = 0; f < num_factors; f++)
            {
                float w_uf = user_factors[user_id, f];
                float h_if = item_factors[item_id, f];
                float h_jf = item_factors[other_item_id, f];

                if (update_u)
                {
                    double update = (h_if - h_jf) * one_over_one_plus_ex - reg_u * w_uf;
                    user_factors[user_id, f] = (float) (w_uf + learn_rate*update);
                }

                if (update_i)
                {
                    double update = w_uf * one_over_one_plus_ex - reg_i * h_if;
                    item_factors[item_id, f] = (float)(h_if + learn_rate * update);
                }

                if (update_j)
                {
                    double update = -w_uf * one_over_one_plus_ex - reg_j * h_jf;
                    item_factors[other_item_id, f] = (float)(h_jf + learn_rate * update);
                }
            }
        }

    }
}
