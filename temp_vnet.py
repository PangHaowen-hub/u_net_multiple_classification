VNet(
  (in_tr): InputTransition(
    (conv1): Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (bn1): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
  )
  (down_tr32): DownTransition(
    (down_conv): Conv3d(16, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (down_tr64): DownTransition(
    (down_conv): Conv3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (1): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (down_tr128): DownTransition(
    (down_conv): Conv3d(64, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (do1): Dropout3d(p=0.5, inplace=False)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (1): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (2): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (down_tr256): DownTransition(
    (down_conv): Conv3d(128, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (do1): Dropout3d(p=0.5, inplace=False)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (1): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (up_tr256): UpTransition(
    (up_conv): ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (do2): Dropout3d(p=0.5, inplace=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (do1): Dropout3d(p=0.5, inplace=False)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (1): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (up_tr128): UpTransition(
    (up_conv): ConvTranspose3d(256, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (do2): Dropout3d(p=0.5, inplace=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (do1): Dropout3d(p=0.5, inplace=False)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (1): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (up_tr64): UpTransition(
    (up_conv): ConvTranspose3d(128, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (do2): Dropout3d(p=0.5, inplace=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (up_tr32): UpTransition(
    (up_conv): ConvTranspose3d(64, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (bn1): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (do2): Dropout3d(p=0.5, inplace=False)
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
    (relu2): LeakyReLU(negative_slope=0.01, inplace=True)
    (ops): Sequential(
      (0): LUConv(
        (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
        (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (out_tr): OutputTransition(
    (conv1): Conv3d(32, 2, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
    (bn1): InstanceNorm3d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (conv2): Conv3d(2, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (relu1): LeakyReLU(negative_slope=0.01, inplace=True)
  )
)