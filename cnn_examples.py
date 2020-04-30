#keras
def build_dqn():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same',
                     input_shape=(4, 80, 80), data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_first'))
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same',
                     data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same',
                     data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same',
                     data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(6))
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model

#pytorch 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(5 * 5 * 64, 512) #1600
        self.fc2 = nn.Linear(512, 6)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 5 * 5 * 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
