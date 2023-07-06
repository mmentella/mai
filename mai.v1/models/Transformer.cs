using mai.v1.blas;

namespace mai.v1.models;

public class Transformer
{
    public static readonly int PositionalEncodingLength = 10000;
    private Matrix positionalEncoding;

    public Transformer(int embeddingSpaceLength, int sequenceLength, string name = null!)
    {
        Name = name.Null() ?? Identity.Next();
        EmbeddingSpaceLength = embeddingSpaceLength;
        SequenceLength = sequenceLength;

        positionalEncoding = new(sequenceLength, embeddingSpaceLength);
    }

    public string Name { get; protected set; }
    public int EmbeddingSpaceLength { get; }
    public int SequenceLength { get; }
}
