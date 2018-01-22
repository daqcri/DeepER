using System;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
namespace PrepData
{
	class Sampler
	{
		public static void Main (string[] args)
		{
			int argsCounter = 0;
			string positivePairsFile = args[argsCounter++];
			string firstFile = args[argsCounter++];
			string secondFile = args[argsCounter++];

			string trainFile = args[argsCounter++];
			string devFile = args[argsCounter++];
			string testFile = args[argsCounter++];


			double trainRatio = double.Parse(args[argsCounter++]);
			double devRatio = double.Parse(args[argsCounter++]);
			double testRatio = double.Parse(args[argsCounter++]);
			int negativeToPositiveRatio = int.Parse(args[argsCounter++]);
			int seed = int.Parse(args[argsCounter++]);





			string line = string.Empty;
			var positivePairs = new List<string> ();
			using (StreamReader reader = new StreamReader (positivePairsFile)) {
				reader.ReadLine (); //skip header
				while ((line = reader.ReadLine ()) != null) {
					positivePairs.Add (line);
				}
			}

			var firstIds = new List<string> ();
			var secondIds = new List<string> ();
			var firstData = new Dictionary<string,string> ();
			var secondData = new Dictionary<string,string> ();

			using (StreamReader firstReader = new StreamReader (firstFile)) {
				var header = firstReader.ReadLine (); //skip header
				var numFields = header.Split(',').Length;
				var firstSentence = new StringBuilder ();
				while ((line = firstReader.ReadLine ()) != null) {
					var splits = line.Split (',');
					var firstId = splits [0];
					firstIds.Add (firstId);
					for (int i = 1; i < splits.Length; i++) {
						firstSentence.Append (splits [i]);
						firstSentence.Append (" ");
					}
					firstSentence.Remove (firstSentence.Length - 1, 1);
					firstData.Add (firstId, firstSentence.ToString());
					firstSentence.Clear ();
				}
			}
			using (StreamReader secondReader = new StreamReader (secondFile)) {
				var header = secondReader.ReadLine (); //skip header
				var numFields = header.Split(',').Length;
				var secondSentence = new StringBuilder ();
				while ((line = secondReader.ReadLine ()) != null) {
					var splits = line.Split (',');
					var secondId = splits [0];
					secondIds.Add (secondId);
					for (int i = 1; i < splits.Length; i++) {
						secondSentence.Append (splits [i]);
						secondSentence.Append (" ");
					}
					secondSentence.Remove (secondSentence.Length - 1, 1);
					secondData.Add (secondId, secondSentence.ToString());
					secondSentence.Clear ();
				}
			}

			var negativePairs = new List<string> ();
			var counter = 0;
			foreach(var positivePair in positivePairs) {
				var ids = positivePair.Split (',');
				var firstId = ids [0];
				var secondId = ids [1];
				counter++;
				var numGotten = 0;
				int negGenSeed = 1;
				var negPairSecondIdsIndices = new HashSet<int> ();
				while (numGotten < negativeToPositiveRatio) {
					Random rNegPair = new Random (counter + negGenSeed++);
					int negPairSecondIdIndex = rNegPair.Next(0, secondIds.Count);
					if (secondIds[negPairSecondIdIndex] != secondId && !negPairSecondIdsIndices.Contains (negPairSecondIdIndex))
					{
						negativePairs.Add (firstId + ',' + secondIds[negPairSecondIdIndex]);
						numGotten++;
					}
				}
			}
				
			var r = new Random (seed);
			positivePairs = positivePairs.OrderBy (item => r.Next ()).ToList<string> ();
			negativePairs = negativePairs.OrderBy (item => r.Next ()).ToList<string> ();

			var numPositivePairsDev = Math.Floor(devRatio * positivePairs.Count);
			var numPositivePairsTesting = Math.Floor(testRatio * positivePairs.Count);
			var numPositivePairsTraining = Math.Floor(trainRatio * positivePairs.Count);

			var numNegativePairsDev = Math.Floor(devRatio * negativePairs.Count);
			var numNegativePairsTesting = Math.Floor(testRatio * negativePairs.Count);
			var numNegativePairsTraining = Math.Floor(trainRatio * negativePairs.Count);


			int writeCounter = 0;
			using (StreamWriter writer = new StreamWriter (devFile)) {
				for (int i = 0; i < numPositivePairsDev; i++) {
					var ids = positivePairs [writeCounter++].Split (',');
					var firstId = ids [0];
					var secondId = ids [1];
					var pairId = string.Join ("_",ids);
					var dataLine = new StringBuilder();
					var firstSentence = firstData [firstId];
					var secondSentence = secondData [secondId];
					dataLine.Append (pairId);
					dataLine.Append (",");
					dataLine.Append (firstId);
					dataLine.Append (",");
					dataLine.Append (secondId);
					dataLine.Append (",");
					dataLine.Append(firstSentence);
					dataLine.Append (",");
					dataLine.Append (secondSentence);
					dataLine.Append (",");
					dataLine.Append ("1");
					writer.WriteLine (dataLine.ToString());
					dataLine.Clear ();
				}
			}
			using (StreamWriter writer = new StreamWriter (testFile)) {
				for (int i = 0; i < numPositivePairsTesting; i++) {
					var ids = positivePairs [writeCounter++].Split (',');
					var firstId = ids [0];
					var secondId = ids [1];
					var pairId = string.Join ("_", ids);
					var dataLine = new StringBuilder();
					var firstSentence = firstData [firstId];
					var secondSentence = secondData [secondId];
					dataLine.Append (pairId);
					dataLine.Append (",");
					dataLine.Append (firstId);
					dataLine.Append (",");
					dataLine.Append (secondId);
					dataLine.Append (",");
					dataLine.Append(firstSentence);
					dataLine.Append (",");
					dataLine.Append (secondSentence);
					dataLine.Append (",");
					dataLine.Append ("1");
					writer.WriteLine (dataLine.ToString());
				}
			}
			using (StreamWriter writer = new StreamWriter (trainFile)) {
				for (int i = 0; i < numPositivePairsTraining; i++) {
					var ids = positivePairs[writeCounter++].Split (',');
					var firstId = ids [0];
					var secondId = ids [1];
					var pairId = string.Join ("_", ids);
					var dataLine = new StringBuilder();
					var firstSentence = firstData [firstId];
					var secondSentence = secondData [secondId];
					dataLine.Append (pairId);
					dataLine.Append (",");
					dataLine.Append (firstId);
					dataLine.Append (",");
					dataLine.Append (secondId);
					dataLine.Append (",");
					dataLine.Append(firstSentence);
					dataLine.Append (",");
					dataLine.Append (secondSentence);
					dataLine.Append (",");
					dataLine.Append ("1");
					writer.WriteLine (dataLine.ToString());
				}
			}
			writeCounter = 0;
			using (StreamWriter writer = new StreamWriter (devFile,true)) {
				for (int i = 0; i < numNegativePairsDev; i++) {
					var ids = negativePairs[writeCounter++].Split (',');
					var firstId = ids [0];
					var secondId = ids [1];
					var pairId = string.Join ("_", ids);
					var dataLine = new StringBuilder();
					var firstSentence = firstData [firstId];
					var secondSentence = secondData [secondId];
					dataLine.Append (pairId);
					dataLine.Append (",");
					dataLine.Append (firstId);
					dataLine.Append (",");
					dataLine.Append (secondId);
					dataLine.Append (",");
					dataLine.Append(firstSentence);
					dataLine.Append (",");
					dataLine.Append (secondSentence);
					dataLine.Append (",");
					dataLine.Append ("2");
					writer.WriteLine (dataLine.ToString());
				}
			}
			using (StreamWriter writer = new StreamWriter (testFile,true)) {
				for (int i = 0; i < numNegativePairsTesting; i++) {
					var ids = negativePairs[writeCounter++].Split (',');
					var firstId = ids [0];
					var secondId = ids [1];
					var pairId = string.Join ("_", ids);
					var dataLine = new StringBuilder();
					var firstSentence = firstData [firstId];
					var secondSentence = secondData [secondId];
					dataLine.Append (pairId);
					dataLine.Append (",");
					dataLine.Append (firstId);
					dataLine.Append (",");
					dataLine.Append (secondId);
					dataLine.Append (",");
					dataLine.Append(firstSentence);
					dataLine.Append (",");
					dataLine.Append (secondSentence);
					dataLine.Append (",");
					dataLine.Append ("2");
					writer.WriteLine (dataLine.ToString());
				}
			}
			using (StreamWriter writer = new StreamWriter (trainFile,true)) {
				for (int i = 0; i < numNegativePairsTraining; i++) {
					var ids = negativePairs[writeCounter++].Split (',');
					var firstId = ids [0];
					var secondId = ids [1];
					var pairId = string.Join ("_",ids);
					var dataLine = new StringBuilder();
					var firstSentence = firstData [firstId];
					var secondSentence = secondData [secondId];
					dataLine.Append (pairId);
					dataLine.Append (",");
					dataLine.Append (firstId);
					dataLine.Append (",");
					dataLine.Append (secondId);
					dataLine.Append (",");
					dataLine.Append(firstSentence);
					dataLine.Append (",");
					dataLine.Append (secondSentence);
					dataLine.Append (",");
					dataLine.Append ("2");
					writer.WriteLine (dataLine.ToString());
				}
			}
		}
	}
}

