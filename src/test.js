import { MnistData } from './data.js';

async function showExamples(data) {
    // Create a container in the visor
    const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' });

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

function getModel() {
    let model = tf.sequential();

    //畳み込みニューラルネットの設定
    model = settingCNN(model);

    //モデルの誤差関数の設定と、学習率の設定
    const optimizer = tf.train.adam(); //学習率の自動調整アルゴリズムの１つ
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy', //クロスエントロピー誤差関数
        metrics: ['accuracy'], //評価指標:精度
    });

    return model;
}

/**
 * 畳み込みニューラルネットの設定
 * @param model モデル
 */
function settingCNN(model) {

    /**学習データのwidth x height と カラーチャネルの指定 */
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    //1層目:畳み込み層
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], //入力データの形状
        kernelSize: 5, //カーネルサイズ
        filters: 8, //フィルター数
        strides: 1, //
        activation: 'relu', //活性化関数
        kernelInitializer: 'varianceScaling'
    }));

    //2層目:プーリング層 : 特徴量の次元削減
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    //3層目:もう一度畳み込み層
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    //４層目：高次元データを1次元に平坦化
    model.add(tf.layers.flatten());

    //5層目：全結合層（出力層）　
    const NUM_OUTPUT_CLASSES = 10; //出力層のノード数(今回は0~9の数字を判別するので10)
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES, //出力層のノード数
        kernelInitializer: 'varianceScaling',
        activation: 'softmax' //活性化函数としてsoftmaxを使用
    }));

    return model;
}

async function train(model, data) {
    /**モニタリングする指標を決定 */
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc']; 
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

/**
 * 予測を行う
 * @param {*} model 
 * @param {*} data 
 * @param {*} testDataSize 
 * @returns 
 */
function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels];
}


/***クラス（数字）ごとの精度を表示 */
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}

async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

    labels.dispose();
}



async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    //学習の実行
    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model); //学習状況の可視化
    await train(model, data);

    await showAccuracy(model, data);
    await showConfusion(model, data);

    console.log('Done', model);
}

document.addEventListener('DOMContentLoaded', run);