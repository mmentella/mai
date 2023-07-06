namespace mai.autograd;

public class Value 
    : IComparable<Value>
{
    private Action? backward;

    public Value(double data, IEnumerable<Value>? children = null, string op = "")
    {
        Data = data;
        Operation = op;
        Children = children ?? Enumerable.Empty<Value>();
        Gradient = 0;
    }

    public double Data { get; set; }

    public string? Label { get; set; }

    public double Gradient { get; private set; }

    public string Operation { get; private set; }

    public IEnumerable<Value> Children { get; private set; }

    public static implicit operator Value(int other) => new(other);
    public static implicit operator Value(double other) => new(other);
    public static implicit operator Value(float other) => new(other);

    public static Value operator +(Value left, Value right)
    {
        var result = new Value(left.Data + right.Data, new[] { left, right }, "+");
        result.backward = () =>
        {
            left.Gradient += result.Gradient;
            right.Gradient += result.Gradient;
        };

        return result;
    }

    public static Value operator *(Value left, Value right)
    {
        var result = new Value(left.Data * right.Data, new[] { left, right }, "*");
        result.backward = () =>
        {
            left.Gradient += right.Data * result.Gradient;
            right.Gradient += left.Data * result.Gradient;
        };

        return result;
    }

    public static Value operator -(Value val) => val * (-1);

    public static Value operator -(Value left, Value right) => left + (-right);

    public static Value operator /(Value left, Value right) => left * right.Pow(-1);

    public Value Exp()
    {
        var result = new Value(Math.Exp(Data), new[] { this }, "exp");
        result.backward = () =>
        {
            Gradient += result.Data * result.Gradient;
        };

        return result;
    }

    public Value Pow(int power)
    {
        var result = new Value(Math.Pow(Data, power), new[] { this }, $"pow({power})");
        result.backward = () =>
        {
            Gradient += power * Math.Pow(Data, power - 1) * result.Gradient;
        };

        return result;
    }

    public Value Tanh()
    {
        var result = new Value(Math.Tanh(Data), new[] { this }, "tanh");
        result.backward = () =>
        {
            Gradient += (1 - Math.Pow(result.Data, 2)) * result.Gradient;
        };

        return result;
    }

    public Value RelU()
    {
        var result = new Value(Math.Max(0, Data), new[] { this }, "relu");
        result.backward = () =>
        {
            Gradient += (Data > 0 ? 1 : 0) * result.Gradient;
        };

        return result;
    }


    public void Backward()
    {
        var topology = new List<Value>();
        var visited = new HashSet<Value>();

        BuildTopology(this);

        Gradient = 1;

        foreach (var val in Enumerable.Reverse(topology))
        {
            val.backward?.Invoke();
        }

        void BuildTopology(Value value)
        {
            if (visited.Contains(value))
            {
                return;
            }

            visited.Add(value);
            foreach (var child in value.Children)
            {
                BuildTopology(child);
            }

            topology.Add(value);
        }
    }

    public void ZeroGrad()
    {
        Gradient = 0;
    }

    public override string ToString()
    {
        return $"value={Data}";
    }
    public int CompareTo(Value? other) => Data.CompareTo(other?.Data);
}
