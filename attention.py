import torch.nn.functional as F
import torch
from torch.nn import Linear, Module, Sequential, Tanh


class FeatureSelfAttention(Module):
    """
    FeatureSelfAttention: An attention mechanism that computes attention weights for each feature in a sequence.
    """

    def __init__(self, embedding_size):
        super(FeatureSelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.attention = Sequential(
            Linear(embedding_size, embedding_size // 2),
            Tanh(),
            Linear(embedding_size // 2, embedding_size),
        )

    def forward(self, x):
        # x shape: (batch_size, embedding_size)

        # Compute attention scores
        attn_scores = self.attention(x)  # (batch_size, embedding_size)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, embedding_size)

        # Apply attention weights to input
        attended_features = x * attn_weights  # (batch_size, embedding_size)

        return attended_features, attn_weights


class CrossAttention(Module):
    def __init__(self, main_feature_size, context_feature_size, hidden_size=64):
        super(CrossAttention, self).__init__()
        self.main_projection = Linear(main_feature_size, hidden_size)
        self.context_projection = Linear(context_feature_size, hidden_size)
        self.attention = Linear(hidden_size, 1)
        self.output_projection = Linear(context_feature_size, main_feature_size)

    def forward(self, main_features, context_features=None):
        # main_features shape: (batch_size, main_feature_size)
        # context_features shape: (batch_size, context_feature_size) or None
        
        if context_features is None:
            return main_features, None
        
        batch_size = main_features.size(0)
        
        main_proj = self.main_projection(main_features)
        context_proj = self.context_projection(context_features)
        
        combined = main_proj.unsqueeze(1) + context_proj.unsqueeze(0)
        
        attn_scores = self.attention(torch.tanh(combined)).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        attended_features = torch.matmul(attn_weights, context_features)
        attended_features_projected = self.output_projection(attended_features)
        
        output_features = main_features + attended_features_projected
        
        return output_features, attn_weights

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 4
    embedding_size = 8

    # 1. Create a small, synthetic dataset
    fake_embeddings = torch.randn(batch_size, embedding_size)
    print("Input embeddings:")
    print(fake_embeddings)
    print("\nInput shape:", fake_embeddings.shape)

    # 2. Initialize your model
    attention_layer = FeatureSelfAttention(embedding_size)

    # 3. Perform a forward pass
    with torch.no_grad():  # We don't need gradients for this test
        attended_features, attention_weights = attention_layer(fake_embeddings)

    # 4. Analyze the outputs
    print("\nAttention weights:")
    print(attention_weights)
    print("\nAttention weights shape:", attention_weights.shape)

    print("\nAttended features:")
    print(attended_features)
    print("\nAttended features shape:", attended_features.shape)

    # Additional analysis
    print("\nSum of attention weights per embedding:")
    print(attention_weights.sum(dim=1))

    print("\nMost attended feature per embedding:")
    print(attention_weights.argmax(dim=1))

    print("\nLeast attended feature per embedding:")
    print(attention_weights.argmin(dim=1))

    # Compare original vs attended features
    print("\nOriginal vs Attended features (first embedding):")
    for i in range(embedding_size):
        print(
            f"Feature {i}: {fake_embeddings[0][i]:.4f} -> {attended_features[0][i]:.4f}"
        )

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 4
    main_feature_size = 8
    context_feature_size = 6

    # Create synthetic data
    main_features = torch.randn(batch_size, main_feature_size)
    context_features = torch.randn(batch_size, context_feature_size)

    print("Main features:")
    print(main_features)
    print("\nMain features shape:", main_features.shape)

    print("\nContext features:")
    print(context_features)
    print("\nContext features shape:", context_features.shape)

    # Initialize the CrossAttention module
    cross_attention = CrossAttention(main_feature_size, context_feature_size)

    # Perform a forward pass
    with torch.no_grad():
        output_features, attention_weights = cross_attention(
            main_features, context_features
        )

    # Analyze the outputs
    print("\nAttention weights:")
    print(attention_weights)
    print("\nAttention weights shape:", attention_weights.shape)

    print("\nOutput features:")
    print(output_features)
    print("\nOutput features shape:", output_features.shape)

    # Additional analysis
    print("\nSum of attention weights per main feature:")
    print(attention_weights.sum(dim=1))

    print("\nMost attended context item for each main item:")
    print(attention_weights.argmax(dim=1))

    print("\nLeast attended context item for each main item:")
    print(attention_weights.argmin(dim=1))

    # Compare original vs output features
    print("\nOriginal vs Output features (first item):")
    for i in range(main_feature_size):
        print(f"Feature {i}: {main_features[0][i]:.4f} -> {output_features[0][i]:.4f}")

    # Check if output features are different from input features
    print("\nAre output features different from input features?")
    print(torch.any(main_features != output_features))

    # Compute and print the difference
    diff = output_features - main_features
    print("\nDifference between output and input features (first item):")
    print(diff[0])

    # Check if attention mechanism is attending to the right dimension
    assert attention_weights.shape == (
        batch_size,
        batch_size,
    ), "Attention weights shape is incorrect"
    print("\nAttention mechanism is attending to the correct dimension.")
