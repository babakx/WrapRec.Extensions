using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WrapRec.Core;
using WrapRec.IO;

namespace WrapRec.Extensions.IO
{
	public class MultiLevelFeedbackCsvReader : CsvReader
	{
		public int Level { get; private set; }

		public override void Setup()
		{
			base.Setup();
			Level = int.Parse(SetupParameters["level"]);
		}

		protected override void EnrichFeedback(Feedback feedback)
		{
			if (feedback.Level == 0)
				feedback.Level = Level;
			if (Level < 0)
				feedback.FeedbackType = FeedbackType.Negative;
		}
	}
}
