using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Net;
using System.Text.RegularExpressions;
using HtmlAgilityPack;
using Newtonsoft.Json;
using PandaLib.text;

namespace DataPreprocess
{
    class Program
    {
        public static readonly string DataPath = @"D:\douban.page.tier2.txt";
        public static readonly string DataFolder = @"D:\";

        public static string DecodeCosmos(string line)
        {
            line = line.Replace("#R#", "\r");
            line = line.Replace("#N#", "\n");
            line = line.Replace("#TAB#", "\t");
            line = line.Replace("#NULL#", "null");
            line = line.Replace("#HASH#", "#");
            return line;
        }

        static void Sample()
        {
            int reviewCnt = 0, commentCnt = 0;
            List<string> reviews = new List<string>();
            List<string> comments = new List<string>();
            using (StreamReader sr = File.OpenText(DataPath))
            {
                while (!sr.EndOfStream && reviewCnt < 2000 && commentCnt < 2000)
                {
                    var line = sr.ReadLine();
                    var splited = line.Split('\t');
                    if (splited[0].Equals("True"))
                    {
                        commentCnt++;
                        comments.Add(line);
                        var body = DecodeCosmos(splited[2]);
                    }
                    else if (splited[0].Equals("False"))
                    {
                        reviewCnt++;
                        reviews.Add(line);
                        var body = DecodeCosmos(splited[2]);
                    }
                }
            }

            using (StreamWriter sw = File.CreateText(DataFolder + "Comment_2000.txt"))
            {
                foreach (var comment in comments)
                {
                    sw.WriteLine(comment);
                }
            }
            using (StreamWriter sw = File.CreateText(DataFolder + "Review_2000.txt"))
            {
                foreach (var review in reviews)
                {
                    sw.WriteLine(review);
                }
            }
        }

        static void DoStatisticsRaw()
        {
            int commentCnt = 0, reviewCnt = 0;
            using (StreamReader sr = File.OpenText(DataPath))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var splited = line.Split('\t');
                    if (splited[0].Equals("True"))
                    {
                        commentCnt++;
                    }
                    else if (splited[0].Equals("False"))
                    {
                        reviewCnt++;
                    }
                }
            }
            Console.WriteLine(string.Format("CommentCnt: {0}, ReviewCnt: {1}", commentCnt, reviewCnt));
            using (StreamWriter sw = File.CreateText(DataFolder + "statistic.txt"))
            {
                sw.WriteLine(string.Format("CommentCnt: {0}, ReviewCnt: {1}", commentCnt, reviewCnt));
            }
        }

        static List<Comment> ParseCommentPage(string htmlText, string url)
        {
            List<Comment> ret = new List<Comment>();
            HtmlDocument doc = new HtmlDocument();
            doc.LoadHtml(htmlText);

            // get movie name
            var contentNode = doc.DocumentNode.SelectSingleNode("//div[@id=\"content\"]");
            var movieName = contentNode.SelectSingleNode("//h1[1]").InnerText;
            movieName = movieName.Substring(0, movieName.Length - 3);

            // get movie id
            var movieIdMatch = Regex.Match(url, @"\d{6,9}");
            var movieId = movieIdMatch.Value;

            foreach (var commentItemNode in doc.DocumentNode.SelectNodes("//div[@class=\"comment-item\"]"))
            {
                var divText = commentItemNode.InnerText;
                if (!string.IsNullOrEmpty(divText) && divText.Trim().Equals("还没有人写过短评"))
                {
                    continue;
                }
                var cid = commentItemNode.GetAttributeValue("data-cid", "0");
                var username = commentItemNode.SelectSingleNode("div[@class=\"avatar\"]/a")
                    .GetAttributeValue("title", "unknown");
                var voteNode = commentItemNode.SelectSingleNode("div[@class=\"comment\"]/h3/span[1]/span[1]");
                var vote = voteNode == null ? "unknown" : voteNode.InnerText;
                var text = commentItemNode.SelectSingleNode("div[@class=\"comment\"]/p").InnerText;
                var rateNode =
                    commentItemNode.SelectSingleNode(
                        "div[@class=\"comment\"]/h3/span[@class=\"comment-info\"]/span[2]");
                string rate;
                if (rateNode != null)
                    rate = rateNode.GetAttributeValue("class", "unknown");
                else
                    rate = "unknown";
                int temp;
                ret.Add(new Comment()
                {
                    Vote = int.TryParse(vote, out temp) ? int.Parse(vote) : 0,
                    Cid = cid,
                    MovieId = movieId,
                    MovieName = movieName,
                    Rate = rate,
                    Text = text,
                    Username = username
                });
            }

            return ret;
        }

        static Review ParseReviewPage(string htmlText, string url)
        {
            HtmlDocument doc = new HtmlDocument();
            doc.LoadHtml(htmlText);

            // get meta information
            var headerNode = doc.DocumentNode.SelectSingleNode("//header[@class=\"main-hd\"]");
            if (headerNode == null)
                return null;
            var username = headerNode.SelectSingleNode("a/span[@property=\"v:reviewer\"]").InnerText;
            var movieUrl = headerNode.SelectSingleNode("a[2]").GetAttributeValue("href", "00000000");
            var movieIdMatch = Regex.Match(movieUrl, @"\d{6,9}");
            var movieId = movieIdMatch.Value;
            var movieName = headerNode.SelectSingleNode("a[2]").InnerText;
            var title = doc.DocumentNode.SelectSingleNode("//head/title").InnerText;
            var reviewIdMatch = Regex.Match(url, @"\d{6,9}");
            var rid = reviewIdMatch.Value;
            string rating;
            var ratingNode = headerNode.SelectSingleNode("span[1]");
            if (ratingNode != null)
                rating = ratingNode.GetAttributeValue("class", "unknown");
            else
                rating = "unknown";
            var text = doc.DocumentNode.SelectSingleNode("//div[@id=\"link-report\"]/div").InnerText;

            return new Review()
            {
                MovieId = movieId,
                MovieName = movieName,
                Rate = rating,
                Rid = rid,
                Text = text,
                Title = title,
                Username = username
            };
        }


        static void GetAllComment()
        {
            int commentCnt = 0;
            HashSet<string> movieIdSet = new HashSet<string>();
            using (StreamReader sr = File.OpenText(DataPath))
            using (StreamWriter sw = File.CreateText(DataFolder + @"AllComments.txt"))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var splited = line.Split('\t');
                    if (splited[0].Equals("True"))
                    {
                        var body = DecodeCosmos(splited[2]);
                        var comments = ParseCommentPage(body, splited[1]);
                        foreach (var comment in comments)
                        {
                            sw.WriteLine(JsonConvert.SerializeObject(comment));
                            commentCnt++;
                            movieIdSet.Add(comment.MovieId);
                        }
                    }
                }
            }
            File.WriteAllText(@"D:\CommentsCnt.txt", commentCnt + "\t" + movieIdSet.Count);
        }

        static void GetReivews(int num)
        {
            HashSet<string> movieIdSet = new HashSet<string>();
            int cnt = 0;
            using (StreamReader sr = File.OpenText(DataPath))
            using (StreamWriter sw = File.CreateText(DataFolder + @"AllReviews.txt"))
            {
                while (!sr.EndOfStream && cnt < num)
                {
                    cnt++;
                    var line = sr.ReadLine();
                    var splited = line.Split('\t');
                    if (splited[0].Equals("False"))
                    {
                        var body = DecodeCosmos(splited[2]);
                        var review = ParseReviewPage(body, splited[1]);
                        if (review == null)
                        {
                            cnt--;
                            continue;
                        }
                        sw.WriteLine(JsonConvert.SerializeObject(review));
                        movieIdSet.Add(review.MovieId);
                    }
                }
            }
            File.WriteAllText(@"D:\ReviewsCnt.txt", num + "\t" + movieIdSet.Count);
        }

        static void CleanData()
        {
            using (StreamReader sr = File.OpenText(DataFolder + @"AllComments.txt"))
            using (StreamWriter sw = File.CreateText(DataFolder + "AllComments.cleaned.txt"))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var comment = JsonConvert.DeserializeObject<Comment>(line);
                    comment.Text = string.Concat(comment.Text.Trim(' ', '\n', '\t')
                        .ToCharArray()
                        .ToList()
                        .Select(t => char.IsPunctuation(t) ? ' ' : t));
                    sw.WriteLine(JsonConvert.SerializeObject(comment));
                }
            }

            using (StreamReader sr = File.OpenText(DataFolder + @"AllReviews.txt"))
            using (StreamWriter sw = File.CreateText(DataFolder + "AllReviews.cleaned.txt"))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var review = JsonConvert.DeserializeObject<Review>(line);
                    review.Text = string.Concat(review.Text.Trim(' ', '\n', '\t')
                        .ToCharArray()
                        .ToList()
                        .Select(t => char.IsPunctuation(t) ? ' ' : t));
                    sw.WriteLine(JsonConvert.SerializeObject(review));
                }
            }
        }

        static void GetRawTextData(bool isCleaned = true, bool addSeparator = false)
        {
            string cleanedStr = isCleaned ? ".cleaned" : "";
            string sepStr = isCleaned ? ".sep" : "";
            using (StreamReader sr = File.OpenText(DataFolder + $@"AllComments{cleanedStr}.txt"))
            using (StreamWriter sw = File.CreateText(DataFolder + $@"AllComments.raw{sepStr}.txt"))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var comment = JsonConvert.DeserializeObject<Comment>(line);
                    var text = comment.Text.Trim();
                    if (addSeparator)
                    {
                        text = text.Replace("#", " ");
                        text += " #";
                    }
                    sw.WriteLine(text);
                }
            }
            using (StreamReader sr = File.OpenText(DataFolder + $@"AllReviews{cleanedStr}.txt"))
            using (StreamWriter sw = File.CreateText(DataFolder + $@"AllReviews.raw{sepStr}.txt"))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var review = JsonConvert.DeserializeObject<Review>(line);
                    var text = review.Text.Trim();
                    if (addSeparator)
                    {
                        text = text.Replace("#", " ");
                        text += " #";
                    }
                    sw.WriteLine(text);
                }
            }
        }

        static void RestoreSegmented()
        {
            using (StreamReader sr = File.OpenText(DataFolder + @"AllComments.raw.sep.segmented.txt"))
            using (StreamReader srJson = File.OpenText(DataFolder + @"AllComments.txt"))
            using (StreamWriter sw = File.CreateText(DataFolder + $@"AllComments.segmented.txt"))
            {
                var lines = sr.ReadLines('#');
                foreach (var line in lines)
                {
                    var commentStr = srJson.ReadLine();
                    var comment = JsonConvert.DeserializeObject<Comment>(commentStr);
                    comment.Text = line.Replace("#", " ").Trim();
                    sw.WriteLine(JsonConvert.SerializeObject(comment));
                }
            }
            using (StreamReader sr = File.OpenText(DataFolder + @"AllReviews.raw.sep.segmented.txt"))
            using (StreamReader srJson = File.OpenText(DataFolder + @"AllReviews.txt"))
            using (StreamWriter sw = File.CreateText(DataFolder + $@"AllReviews.segmented.txt"))
            {
                var lines = sr.ReadLines('#');
                foreach (var line in lines)
                {
                    var commentStr = srJson.ReadLine();
                    var review = JsonConvert.DeserializeObject<Review>(commentStr);
                    review.Text = line.Replace("#", " ").Trim();
                    sw.WriteLine(JsonConvert.SerializeObject(review));
                }
            }
        }

        static Dictionary<string, int> VocCount(string filename)
        {
            Dictionary<string, int> dict = new Dictionary<string, int>();
            using (StreamReader sr = File.OpenText(filename))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var splited = line.Split(' ');
                    foreach (var word in splited)
                    {
                        var trimedWord = word.Trim();
                        if (!dict.ContainsKey(trimedWord))
                        {
                            dict.Add(trimedWord, 0);
                        }
                        dict[trimedWord]++;
                    }
                }
            }
            return dict;
        }

        static Metric DoStatisticsComments(string filename)
        {
            var ret = new Metric();
            var lines = File.ReadLines(DataFolder + filename);
            foreach (var line in lines)
            {
                var comment = JsonConvert.DeserializeObject<Comment>(line);
                if (ret.Ids.Contains(comment.Cid))
                    continue;
                if (!Regex.IsMatch(comment.Rate, @"\d+"))
                    continue;
                ret.Ids.Add(comment.Cid);

                if (!ret.Movies.ContainsKey(comment.MovieId))
                    ret.Movies.Add(comment.MovieId, 0);
                ret.Movies[comment.MovieId]++;

                if (!ret.Users.ContainsKey(comment.Username))
                    ret.Users.Add(comment.Username, 0);
                ret.Users[comment.Username]++;
            }
            return ret;
        }

        static void OutputMetric(Metric metric)
        {
            Console.WriteLine($"All different comments {metric.Ids.Count}");
            Console.WriteLine($"All different movies {metric.Movies.Count}");
            Console.WriteLine($"All different users {metric.Users.Count}");
            Console.WriteLine();

            Console.WriteLine($"All movies more than 3 {metric.Movies.Count(t => t.Value >= 3)}");
            Console.WriteLine($"All users more than 3 {metric.Users.Count(t => t.Value >= 3)}");
            Console.WriteLine($"Commetns for active users 3 {metric.Users.Sum(t => t.Value > 3 ? t.Value : 0)}");
            Console.WriteLine();

            Console.WriteLine($"All movies more than 5 {metric.Movies.Count(t => t.Value >= 5)}");
            Console.WriteLine($"All users more than 5 {metric.Users.Count(t => t.Value >= 5)}");
            Console.WriteLine($"Commetns for active users 3 {metric.Users.Sum(t => t.Value > 5 ? t.Value : 0)}");
            Console.WriteLine();

            Console.WriteLine($"All movies more than 10 {metric.Movies.Count(t => t.Value >= 10)}");
            Console.WriteLine($"All users more than 10 {metric.Users.Count(t => t.Value >= 10)}");
            Console.WriteLine($"Commetns for active users 3 {metric.Users.Sum(t => t.Value > 10 ? t.Value : 0)}");
            Console.WriteLine();
            Console.ReadLine();
        }

        static void GetBetterData(string filename)
        {
            var metric = DoStatisticsComments(filename);
            var validUsers = metric.Users.Where(t => t.Value > 5).ToDictionary(t => t.Key, t => t.Value);
            var lines = File.ReadLines(DataFolder + filename);
            var ids = new HashSet<string>();
            var results = new List<string>();
            int cnt = 0;
            foreach (var line in lines)
            {
                var comment = JsonConvert.DeserializeObject<Comment>(line);
                if (validUsers.ContainsKey(comment.Username))
                {
                    if (ids.Contains(comment.Cid))
                        continue;
                    if (!Regex.IsMatch(comment.Rate, @"\d+"))
                        continue;
                    ids.Add(comment.Cid);
                    results.Add(line);
                    cnt++;
                    if (cnt == 60000)
                        break;
                }
            }
            File.WriteAllLines(DataFolder + @"Comment60k.txt", results);
        }

        static void ShuffleAndDivide(string filename, int trainSize, int devSize, int testSize)
        {
            var lines = File.ReadAllLines(DataFolder + filename);
            if (lines.Length < trainSize + testSize + devSize)
            {
                Console.WriteLine("We don't have so many sentences !");
                return;
            }
            lines = lines.OrderBy(t => Guid.NewGuid()).ToArray();
            File.WriteAllLines(DataFolder + filename + ".train.txt", lines.Take(trainSize));
            File.WriteAllLines(DataFolder + filename + ".dev.txt", lines.Skip(trainSize).Take(devSize));
            File.WriteAllLines(DataFolder + filename + ".test.txt", lines.Skip(trainSize + devSize).Take(testSize));
        }

        static void DivideByCharacter(string filename)
        {
            var lines = File.ReadAllLines(DataFolder + filename);
            var tokenizer = new EnglishTokenizer();
            var results = new List<string>();
            foreach (var line in lines)
            {
                var comment = JsonConvert.DeserializeObject<Comment>(line);
                var text = comment.Text;
                var tokens = tokenizer.Process(text);
                comment.Text = string.Join(" ", tokens.Select(t => text.Substring(t.Start, t.Length)));
                results.Add(JsonConvert.SerializeObject(comment));
            }
            File.WriteAllLines(DataFolder + filename + ".char.txt", results);
        }

        // for DMSC data
        static void GetAllCommentDMSC(string filename)
        {
            var lines = File.ReadLines(filename);
            foreach (var line in lines)
            {
                var splited = line.Split(',');
                // not used
            }
        }


        static void Main(string[] args)
        {
            //GetBetterData("AllComments.segmented.txt");
            //ShuffleAndDivide("Comment60k.txt", 40000, 10000, 10000);
            //OutputMetric(DoStatisticsComments("AllComments.segmented.txt"));
            DivideByCharacter("Comment60k.txt.train.txt");
            DivideByCharacter("Comment60k.txt.test.txt");
            DivideByCharacter("Comment60k.txt.dev.txt");
        }


    }

    static class ExtensionsForTextReader
    {
        public static IEnumerable<string> ReadLines(this TextReader reader, char delimiter)
        {
            List<char> chars = new List<char>();
            while (reader.Peek() >= 0)
            {
                char c = (char)reader.Read();

                if (c == delimiter)
                {
                    yield return new string(chars.ToArray());
                    chars.Clear();
                    continue;
                }

                chars.Add(c);
            }
        }
    }
}
