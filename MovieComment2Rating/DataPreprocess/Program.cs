using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Net;
using System.Text.RegularExpressions;
using HtmlAgilityPack;
using javax.swing.text;
using Newtonsoft.Json;

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

        static void DoStatistics()
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
            File.WriteAllText(@"D:\CommentsCnt.txt", num + "\t" + movieIdSet.Count);
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


        static void Main(string[] args)
        {
            //GetAllComment();
            //GetReivews(100000);
            //CleanData();
            //GetRawTextData(true, true);
            //RestoreSegmented();

            //var dict = VocCount(DataFolder + "AllReviews.txt");
            //Console.WriteLine(dict.Where(t => t.Value > 4).Count());
            //Console.ReadLine();

            DoStatistics();

            //int cnt = 0;
            //using (StreamReader sr = File.OpenText(@"D:\AllComments.txt"))
            //{
            //    while (!sr.EndOfStream)
            //    {
            //        var line = sr.ReadLine();
            //        cnt++;
            //    }
            //}
            //Console.WriteLine(cnt);
            //Console.ReadLine();
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
