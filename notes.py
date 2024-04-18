# PyImageSearch
def get_pixel_value(B, H, W, featureMap, x, y):
    # create batch indices and reshape it
    batchIdx = tf.range(0, B)
    batchIdx = tf.reshape(batchIdx, (B, 1, 1))

    # create the indices matrix which will be used to sample the
    # feature map
    b = tf.tile(batchIdx, (1, H, W))
    indices = tf.stack([b, y, x], 3)

    # gather the feature map values for the corresponding indices
    gatheredPixelValue = tf.gather_nd(featureMap, indices)

    # return the gather pixel values
    return gatheredPixelValue


def affine_grid_generator(B, H, W, theta):
    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, H)
    y = tf.linspace(-1.0, 1.0, W)
    (xT, yT) = tf.meshgrid(x, y)

    # flatten the meshgrid
    xTFlat = tf.reshape(xT, [-1])
    yTFlat = tf.reshape(yT, [-1])

    # reshape the meshgrid and concatenate ones to convert it to
    # homogenous form
    ones = tf.ones_like(xTFlat)
    samplingGrid = tf.stack([xTFlat, yTFlat, ones])

    # repeat grid batch size times
    samplingGrid = tf.broadcast_to(samplingGrid, (B, 3, H * W))

    # cast the affine parameters and sampling grid to float32
    # required for matmul
    theta = tf.cast(theta, "float32")
    samplingGrid = tf.cast(samplingGrid, "float32")

    # transform the sampling grid with the affine parameter
    batchGrids = tf.matmul(theta, samplingGrid)

    # reshape the sampling grid to (B, H, W, 2)
    batchGrids = tf.reshape(batchGrids, [B, 2, H, W])

    # return the transformed grid
    return batchGrids


def bilinear_sampler(B, H, W, featureMap, x, y):
    # define the bounds of the image
    maxY = tf.cast(H - 1, "int32")
    maxX = tf.cast(W - 1, "int32")
    zero = tf.zeros([], dtype="int32")

    # rescale x and y to feature spatial dimensions
    x = tf.cast(x, "float32")
    y = tf.cast(y, "float32")
    x = 0.5 * ((x + 1.0) * tf.cast(maxX - 1, "float32"))
    y = 0.5 * ((y + 1.0) * tf.cast(maxY - 1, "float32"))

    # grab 4 nearest corner points for each (x, y)
    x0 = tf.cast(tf.floor(x), "int32")
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), "int32")
    y1 = y0 + 1

    # clip to range to not violate feature map boundaries
    x0 = tf.clip_by_value(x0, zero, maxX)
    x1 = tf.clip_by_value(x1, zero, maxX)
    y0 = tf.clip_by_value(y0, zero, maxY)
    y1 = tf.clip_by_value(y1, zero, maxY)

    # get pixel value at corner coords
    Ia = get_pixel_value(B, H, W, featureMap, x0, y0)
    Ib = get_pixel_value(B, H, W, featureMap, x0, y1)
    Ic = get_pixel_value(B, H, W, featureMap, x1, y0)
    Id = get_pixel_value(B, H, W, featureMap, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, "float32")
    x1 = tf.cast(x1, "float32")
    y0 = tf.cast(y0, "float32")
    y1 = tf.cast(y1, "float32")

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute transformed feature map
    transformedFeatureMap = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    # return the transformed feature map
    return transformedFeatureMap


class STN(Layer):
    def __init__(self, name, filter):
        # initialise the layer
        super().__init__(name=name)
        self.B = None
        self.H = None
        self.W = None
        self.C = None

        # create the constant bias initializer
        self.output_bias = tf.keras.initializers.Constant(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )

        # define the filter size
        self.filter = filter

    def build(self, input_shape):
        # get the batch size, height, width and channel size of the
        # input
        (self.B, self.H, self.W, self.C) = input_shape

        # define the localizaiton network
        self.localizationNet = Sequential(
            [
                Conv2D(
                    filters=self.filter // 4,
                    kernel_size=3,
                    input_shape=(self.H, self.W, self.C),
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                MaxPool2D(),
                Conv2D(
                    filters=self.filter // 2,
                    kernel_size=3,
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                MaxPool2D(),
                Conv2D(
                    filters=self.filter,
                    kernel_size=3,
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                MaxPool2D(),
                GlobalAveragePooling2D(),
            ]
        )

        # define the regressor network
        self.regressorNet = tf.keras.Sequential(
            [
                Dense(
                    units=self.filter, activation="relu", kernel_initializer="he_normal"
                ),
                Dense(
                    units=self.filter // 2,
                    activation="relu",
                    kernel_initializer="he_normal",
                ),
                Dense(
                    units=3 * 2,
                    kernel_initializer="zeros",
                    bias_initializer=self.output_bias,
                ),
                Reshape(target_shape=(2, 3)),
            ]
        )

    def call(self, x):
        # get the localization feature map
        localFeatureMap = self.localizationNet(x)

        # get the regressed parameters
        theta = self.regressorNet(localFeatureMap)

        # get the transformed meshgrid
        grid = affine_grid_generator(self.B, self.H, self.W, theta)

        # get the x and y coordinates from the transformed meshgrid
        xS = grid[:, 0, :, :]
        yS = grid[:, 1, :, :]

        # get the transformed feature map
        x = bilinear_sampler(self.B, self.H, self.W, x, xS, yS)

        # return the transformed feature map
        return x


def get_training_model(batchSize, height, width, channel, stnLayer, numClasses, filter):
    # define the input layer and pass the input through the STN
    # layer
    inputs = Input((height, width, channel), batch_size=batchSize)
    x = Lambda(lambda image: tf.cast(image, "float32") / 255.0)(inputs)
    x = stnLayer(x)

    # apply a series of conv and maxpool layers
    x = Conv2D(filter // 4, 3, activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPool2D()(x)
    x = Conv2D(filter // 2, 3, activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPool2D()(x)
    x = Conv2D(filter, 3, activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPool2D()(x)

    # global average pool the output of the previous layer
    x = GlobalAveragePooling2D()(x)

    # pass the flattened output through a couple of dense layers
    x = Dense(filter, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(filter // 2, activation="relu", kernel_initializer="he_normal")(x)

    # apply dropout for better regularization
    x = Dropout(0.5)(x)

    # apply softmax to the output for a multi-classification task
    outputs = Dense(numClasses, activation="softmax")(x)

    # return the model
    return Model(inputs, outputs)


# PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
