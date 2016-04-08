using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WrapRec.Core;
using WrapRec.IO;

namespace WrapRec.Extensions.IO
{
	public class RatingAsMultiLevelFeedbackCsvReader : CsvReader
	{
		public int MaxRating { get; set; }

		public override void Setup()
		{
			base.Setup();
			MaxRating = int.Parse(SetupParameters["maxRating"]);
		}
		
		protected override void EnrichFeedback(Feedback feedback)
		{
			feedback.Level = MaxRating - (int) ((Rating)feedback).Value;
		}
	}
}
