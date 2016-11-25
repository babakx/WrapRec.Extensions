using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite;
using MyMediaLite.ItemRecommendation;
using WrapRec.Models;

namespace WrapRec.Extensions.Models
{
    public class WrapRecWRMF : WRMF, IModelAwareRecommender
    {
        public Model Model { get; set; }

        public override void Iterate()
        {
            int time = (int)Wrap.MeasureTime(delegate () { base.Iterate(); }).TotalMilliseconds;
            Model.OnIterate(this, time);
        }
    }
}
