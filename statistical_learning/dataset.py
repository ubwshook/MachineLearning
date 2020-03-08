from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def get_iris_data():
    """
    生成所需要的数据集，这里使用的是iris数据集。
    Iris数据集在模式识别研究领域应该是最知名的数据集了，有很多文章都用到这个数据集。这个数据集里一共包括150行记录，
    其中前四列为花萼长度，花萼宽度，花瓣长度，花瓣宽度等4个用于识别鸢尾花的属性，第5列为鸢尾花的类别（包括Setosa，Versicolour，
    Virginica三类）。也即通过判定花萼长度，花萼宽度，花瓣长度，花瓣宽度的尺寸大小来识别鸢尾花的类别。
    我们使用前100个数据，也就只有前两种分类，同时我们仅使用两种属性即花萼长度、花萼宽度。
    :return: 总训数据集 train_data, 训练数据 X_train, 测试数据 X_test, 训练数据标签 y_train, y_test, 测试数据标签 feature_names
    """
    # 导入iris数据集
    iris = load_iris()

    data = iris.data
    data = data[:100]
    labels = iris.target
    labels = labels[:100]
    feature_names = iris.feature_names
    print("feature_name", feature_names)

    # 我们选择了sepal length (cm)', 'sepal width (cm)'，并且更改了labels
    train_data = data[:, 0:2]
    labels[labels == 0] = -1
    feature_names = feature_names[:2]

    # 划分数据集，75个训练数据，25个测试数据
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels)  # 划分数据集
    return train_data, X_train, X_test, y_train, y_test, feature_names