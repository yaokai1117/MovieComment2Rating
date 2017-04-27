using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPreprocess
{
    public class Page
    {
        public bool IsComment { get; set; }
        public string Url { get; set; }
        public string Body { get; set; }
    }

    public class Comment
    {
        public string Text { get; set; }
        public string Username { get; set; }
        public string Rate { get; set; }
        public string Cid { get; set; }
        public int Vote { get; set; }
        public string MovieName { get; set; }
        public string MovieId { get; set; }
    }

    public class Review
    {
        public string Title { get; set; }
        public string Text { get; set; }
        public string Username { get; set; }
        public string Rate { get; set; }
        public string Rid { get; set; }
        public string MovieName { get; set; }
        public string MovieId { get; set; }
    }

    public class Metric
    {
        public Metric()
        {
            Ids = new HashSet<string>();
            Users = new Dictionary<string, int>();
            Movies = new Dictionary<string, int>();
        }
        public HashSet<string> Ids;
        public Dictionary<string, int> Users;
        public Dictionary<string, int> Movies;
    }
}
