import torch


def test():
    B = 20
    D = 100
    N = 7

    A = torch.rand(B, D)
    B = torch.rand(B, N, D)
    A = A.unsqueeze(1)

    # Dot product along the last axis (dimension D)
    result = torch.sum(A * B, dim=(1, 2))

    print(result.shape)

if __name__ == "__main__":
    test()