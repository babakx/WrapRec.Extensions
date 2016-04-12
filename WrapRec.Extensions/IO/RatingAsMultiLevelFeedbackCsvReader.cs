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
		
		protected override void EnrichFeedback(Feedback feedback)
		{
			feedback.Level = (int) ((Rating)feedback).Value;
		}
	}
}
