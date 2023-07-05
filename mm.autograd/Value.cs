namespace mm.autograd;

public class Value
    : IComparable<Value>
{
    private readonly double data;
    private readonly string? name;
    private readonly Value[] children;
    private readonly double gradient;

    private Action? backward;

    public Value(double value, string? name = null, params Value[] children)
    {
        backward = null;
        this.data = value;
        this.name = name;
        this.children = children;

        gradient = 0;
    }

    public double Data => data;

    public double Gradient => gradient;

    public Value[] Children => children;

    public static implicit operator Value(double value) => new(value);
    public static implicit operator Value(int value) => new(value);
    public static implicit operator Value(float value) => new(value);

    public int CompareTo(Value? other) => data.CompareTo(other?.data);

    public override string ToString()
    {
        return $"value={data}";
    }
}
