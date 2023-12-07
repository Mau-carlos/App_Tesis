package fisei.vasconez.app_tesis.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.media.MediaScannerConnection
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import fisei.vasconez.app_tesis.ml.AlphanumericML
import fisei.vasconez.app_tesis.ml.SsdMobilenet
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Arrays
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.random.Random


class ImageSegmentationHelper
    (
    var currentModel: Int = 0,
    var numThreads: Int = 4,
    val context: Context,
    val classifierListener: ClassifierListener?
    ) {


    private var interpreter_ReUNet: Interpreter? = null
    private var targetWidth_ReUNet: Int = 0
    private var targetHeight_ReUNet: Int = 0

    private var interpreter_ssdmobilenet: SsdMobilenet? = null
    private var targetWidth_ssdmobilenet : Int = 0
    private var targetHeight_ssdmobilenet : Int = 0

    private var interpreter_cnn: Interpreter? = null
    private var targetWidth_cnn : Int = 0
    private var targetHeight_cnn : Int = 0

    /**
     * Constructor de la clase
     */
    init {
        if (setupModelResUNet()) {

            targetWidth_ReUNet = interpreter_ReUNet!!.getInputTensor(0).shape()[2]
            targetHeight_ReUNet = interpreter_ReUNet!!.getInputTensor(0).shape()[1]

        } else {
            classifierListener?.onError("TFLite ResUNet failed to init. ❌")
        }

        if (setupModelSSDMobileNet()) {


            targetWidth_ssdmobilenet = 320
            targetHeight_ssdmobilenet = 320

        } else {
            classifierListener?.onError("TFLite ResUNet failed to init. ❌")
        }

        if (setupCnn()) {


            targetWidth_cnn = 128
            targetHeight_cnn = 128

        } else {
            classifierListener?.onError("TFLite CNN failed to init. ❌")
        }
    }


    /**
     * Funcion Iguala los interpreter a null
     * SSD_Mobile_Net y ResUNet se igualan a null
     */
    fun close() {
        interpreter_ReUNet = null
        interpreter_ssdmobilenet = null
        interpreter_cnn = null
    }


    /**
     * Funcion que inicializa el tflite ResUNet
     * @return Boolean
     * @exception IOException No puede inicializar el modelo
     */
    private fun setupModelResUNet(): Boolean {
        val options = Interpreter.Options()
        options.numThreads = numThreads

        return try {
            val modelFile = FileUtil.loadMappedFile(context, "plateSegmentationML.tflite")
            interpreter_ReUNet = Interpreter(modelFile, options)
            true
        } catch (e: IOException) {
            classifierListener?.onError(
                "Model ResUNet Fallo ❌ " +
                        "initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed ❌ to load model with error: " + e.message)
            false
        }
    }

    /**
     * Funcion inicializa el modelo SSDMobileNet
     * @return Boolean inicializa correctamente el modelo
     * @exception IOException No pudo inicializar el modelo
     */
    private fun setupModelSSDMobileNet(): Boolean {
        val options = Interpreter.Options()
        options.numThreads = numThreads

        return try {
            interpreter_ssdmobilenet = SsdMobilenet.newInstance(context)
            true
        } catch (e: IOException) {
            classifierListener?.onError(
                "Model SSD_MOBILE_NET Fallo ❌ " + "initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite SSD_MOBILE_NET failed ❌ to load model with error: " + e.message)
            false
        }
    }

    /**
     * Funcion inicializa el modelo SSDMobileNet
     * @return Boolean inicializa correctamente el modelo
     * @exception IOException No pudo inicializar el modelo
     */
    private fun setupCnn(): Boolean {
        val options = Interpreter.Options()
        options.numThreads = numThreads

        return try {
            val modelFile = FileUtil.loadMappedFile(context, "alphanumericML.tflite")
            interpreter_cnn = Interpreter(modelFile, options)
            true
        } catch (e: IOException) {
            classifierListener?.onError(
                "Model CNN Fallo ❌ " + "initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite CNN failed ❌ to load model with error: " + e.message)
            false
        }
    }

    /**
     * Funcion realiza la clasificacion
     * @param bitmap Bitmap
     * @param rotation Int
     */
    fun classify(
        bitmap: Bitmap,
        rotation: Int
    ) {
        val copiedBitmap: Bitmap = bitmap.copy(bitmap.config, true)
        val imgCopy = processImage(copiedBitmap, rotation )
        val img2 = imgCopy!!.bitmap

        processInputImage(bitmap, rotation)?.let { image ->
            if (interpreter_ReUNet == null) {
                setupModelResUNet()
            }

            var inferenceTime = SystemClock.uptimeMillis()

            // Crear TensorBuffer para la salida con las dimensiones correctas y el mismo tipo de datos
            val output = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 1), DataType.FLOAT32)


            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(image.buffer)

            interpreter_ReUNet?.run(inputFeature0.buffer, output.buffer )

            val salida = convertByteBufferToBitmap(output.buffer, 256, 256)
//            val random = Random
//            val randomNumber = random.nextInt()
//
//            saveToGallery(salida!!, "segementacion_$randomNumber")

            val result = processOpencv(salida!!, img2)
            var placa:String? = ""
            if(result != null ){
                val caracteres  = detectObjects(result)
                Log.e("🦊🦊", "Esta es placa ${caracteres?.size}")
                if(caracteres != null){
                  placa =  cnnClassify(caracteres)
                }

            }

            inferenceTime = SystemClock.uptimeMillis() - inferenceTime

            classifierListener?.onResults(placa, inferenceTime)

        }
    }


    private fun cnnClassify(listaCaracteres : MutableList<Bitmap>): String?{
        try {
            var placa = ""
            for ((indice, bitmap) in listaCaracteres.withIndex()) {

                if (interpreter_cnn == null) {
                    setupCnn()
                }

                val output = TensorBuffer.createFixedSize(intArrayOf(1, 36), DataType.FLOAT32)
                interpreter_cnn?.run(convertBitmapToByteBuffer(bitmap), output.buffer)
                val posiblesEtiq2 = listOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
                val outputArray: FloatArray = output.floatArray
                Log.e("SALIDA TENSOR", "🚨🚨 ${outputArray.sliceArray(0 until 36).contentToString()}")
                val maxIndex = outputArray.indexOfLast { it == outputArray.maxOrNull() }
                val predictedClassLabel = posiblesEtiq2[maxIndex]
                placa += predictedClassLabel

                Log.e("👑👑👑👑👑 PLACA", "Esta es placa ${placa}")

            }

            return placa

        }catch (e: Exception){
            Log.e(TAG, "Funcion CNN fallo ${e.message}")
        }
        return null
    }


    /**
     * Funcion para detectar los caracteres y los recorta y los devuelve en una MutableList
     * @param imgCrop Bitmap
     * @return MutableList<Bitmap>?
     */
    private fun detectObjects(imgCrop : Bitmap ): MutableList<Bitmap>?{
        try {
            var caracteres  :  MutableList<Bitmap>
            processInputImage_ssdmobilenet(imgCrop)?.let { image ->
                if (interpreter_ssdmobilenet == null) {
                    setupModelSSDMobileNet()
                }

                // Creates inputs for reference.
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 320, 320, 3), DataType.FLOAT32)
                inputFeature0.loadBuffer(image.buffer)


                val result = interpreter_ssdmobilenet?.process(inputFeature0)

                val outputFeature0 = result?.outputFeature0AsTensorBuffer
                val outputFeature1 =  result?.outputFeature1AsTensorBuffer
                val outputFeature2 =  result?.outputFeature2AsTensorBuffer
                val outputFeature3 =  result?.outputFeature3AsTensorBuffer
                caracteres = cropCharacteres2(imgCrop, outputFeature1,outputFeature0, imgCrop.width, imgCrop.height, 0.25f)
                Log.e("🥶🥶🥶🥶", "${caracteres.size}")
                return  caracteres
            }

        }catch (e: Exception){
            Log.e(TAG, "Funcion detectObjects fallo ${e.message}")
        }
         return null
    }


    /**
     * Funcion para recortar cada caracter de la placa
     * @param img Bitmap
     * @param boxesTensor TensorBuffer?
     * @param scoresTensor TensorBuffer?
     * @param width Int
     * @param height Int
     * @param minConfidence Float
     */
//    private fun cropCharacteres2(
//        img: Bitmap,
//        boxesTensor: TensorBuffer?,
//        scoresTensor: TensorBuffer?,
//        width: Int, height: Int,
//        minConfidence: Float
//    ) : MutableList<Bitmap>  {
//
//        val caracteres = mutableListOf<Bitmap>()
//        val boxes: FloatArray? = boxesTensor?.floatArray
//        val scores: FloatArray? = scoresTensor?.floatArray
//
//
//        if ( boxes != null && scores != null) {
//            for (i in scores.indices) {
//                val score = scores[i]
//
//                if (score > minConfidence && score <= 1.0) {
//                    val ymin = boxes[i * 4]
//                    val xmin = boxes[i * 4 + 1]
//                    val ymax = boxes[i * 4 + 2]
//                    val xmax = boxes[i * 4 + 3]
//
//                    // Calcula las coordenadas reales usando la altura y anchura de la imagen
//                    val roi = floatArrayOf(
//                        ymin * height,
//                        xmin * width,
//                        ymax * height,
//                        xmax * width
//                    )
//
//                    // Asegúrate de que las coordenadas están dentro de los límites de la imagen
//                    val clippedROI = floatArrayOf(
//                        roi[0].coerceIn(0f, height.toFloat()),
//                        roi[1].coerceIn(0f, width.toFloat()),
//                        roi[2].coerceIn(0f, height.toFloat()),
//                        roi[3].coerceIn(0f, width.toFloat())
//                    )
//
//                    val region = Bitmap.createBitmap(
//                        img,
//                        clippedROI[1].toInt(),
//                        clippedROI[0].toInt(),
//                        (clippedROI[3] - clippedROI[1]).toInt(),
//                        (clippedROI[2] - clippedROI[0]).toInt()
//                    )
//                    val letra = escalarImagen(region)
////                    val random = Random
////                    val randomNumber = random.nextInt()
////                    Log.e("GALERIA  ", " OKK 👌👌👌")
////                    saveToGallery(letra, "letra_$randomNumber")
////                    Log.e("GALERIA OKK ", " OKK 👌👌👌")
////                    val resizedAndGrayscaleBitmap = resizeAndConvertToGrayscale(letra, 128, 128)
//
//                    caracteres.add(letra)
//                }
//            }
//        }
//        return caracteres
//    }

    private fun cropCharacteres2(
        img: Bitmap,
        boxesTensor: TensorBuffer?,
        scoresTensor: TensorBuffer?,
        width: Int,
        height: Int,
        minConfidence: Float
    ): MutableList<Bitmap> {
        val caracteres = mutableListOf<Bitmap>()

        val boxes: FloatArray? = boxesTensor?.floatArray
        val scores: FloatArray? = scoresTensor?.floatArray

        if (boxes != null && scores != null) {
            val numBoxes = scores.size

            // Crear una lista de índices ordenados de acuerdo al eje X
            val boxesOrdenadasIndices = (0 until numBoxes).sortedBy { boxes[it * 4 + 1] }

            for (i in boxesOrdenadasIndices) {
                val index = i * 4

                if (index + 3 < boxes.size && i < scores.size) {
                    val score = scores[i]

                    if (score > minConfidence && score <= 1.0) {
                        val ymin = boxes[index]
                        val xmin = boxes[index + 1]
                        val ymax = boxes[index + 2]
                        val xmax = boxes[index + 3]

                        // Calcula las coordenadas reales usando la altura y anchura de la imagen
                        val roi = floatArrayOf(
                            ymin * height,
                            xmin * width,
                            ymax * height,
                            xmax * width
                        )

                        // Asegúrate de que las coordenadas están dentro de los límites de la imagen
                        val clippedROI = floatArrayOf(
                            roi[0].coerceIn(0f, height.toFloat()),
                            roi[1].coerceIn(0f, width.toFloat()),
                            roi[2].coerceIn(0f, height.toFloat()),
                            roi[3].coerceIn(0f, width.toFloat())
                        )

                        try {
                            val region = Bitmap.createBitmap(
                                img,
                                clippedROI[1].toInt(),
                                clippedROI[0].toInt(),
                                (clippedROI[3] - clippedROI[1]).toInt(),
                                (clippedROI[2] - clippedROI[0]).toInt()
                            )
                            val letra = escalarImagen(region)
//                            val random = Random
//                            val randomNumber = random.nextInt()
//                            Log.e("GALERIA  ", " OKK 👌👌👌")
//                            saveToGallery(letra, "letra_$randomNumber")
                            caracteres.add(letra)
                        } catch (e: Exception) {
                            Log.e("cropCharacteres2", "Error creando Bitmap", e)
                        }
                    }
                } else {
                    Log.e("cropCharacteres2", "Índice fuera de rango: $i")
                }
            }
        }
        return caracteres
    }


    /**
     * Funcion Prepocesamiento IMG antes de ingresar al modelo tflite ResUNet
     * @param image Bitmap Image para ser preprocesada
     * @param imageRotation angulo de rotacion de la imagen
     * @return TensorImage? retorna una un tensor para el modelo ResUNet
     */
    private fun processInputImage(
        image: Bitmap,
        imageRotation: Int
    ): TensorImage? {
        val height = image.height
        val width = image.width
        val cropSize = min(height, width)
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .add(
                ResizeOp(
                    targetHeight_ReUNet,
                    targetWidth_ReUNet,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            ).add(NormalizeOp(127.5f, 127.5f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }


    /**
     * Funcion dedicada a preprocesar la entrada del Modelos SSD Mobile Net
     * @param image Bitmap
     * @return TensorImage?
     */
    private fun processInputImage_ssdmobilenet(
        image: Bitmap
    ): TensorImage? {
        val imageProcessor = ImageProcessor.Builder()
            .add(
                ResizeOp(
                    targetHeight_ssdmobilenet,
                    targetWidth_ssdmobilenet,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            ).add(NormalizeOp(127.5f, 127.5f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }


    private fun processInputImage_cnn(
        image: Bitmap
    ): TensorImage? {

        val imageProcessor = ImageProcessor.Builder()
            .add(
                ResizeOp(
                    targetHeight_cnn,
                    targetWidth_cnn,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            ).add(NormalizeOp(127.5f, 127.5f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }


    /**
     * Para Tomar solo un gray CNN
     * @param originalImage Bitmap
     * @return Bitmap
     */
    private fun convertToGrayscale(originalImage: Bitmap): Bitmap {
        // Convierte la imagen a escala de grises
        val grayImage = Bitmap.createBitmap(
            originalImage.width,
            originalImage.height,
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(grayImage)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(originalImage, 0f, 0f, paint)
        return grayImage
    }
    /**
     * Funcion para encontrar el contorno de la img segmentada y recorta la region segmentada
     * @param mask Bitmap img mask segmentada
     * @param img Bitmap img original para recortar la rgion de interes ROI
     */
    private fun processOpencv(
        mask : Bitmap,
        img : Bitmap
    ):Bitmap? {

            //MASK
            val mat_mask = bitmaptoMat(mask)
            val width = img.width
            val height = img.height
            val matImageOriginal = Mat(height, width, CvType.CV_8UC3)
            val imageDataOriginal = ByteArray(width * height * 3)
            img.getPixels(
                IntArray(width * height),
                0,
                width,
                0,
                0,
                width,
                height
            )

            for (i in 0 until width * height) {
                val pixelValue = img.getPixel(i % width, i / width)
                imageDataOriginal[i * 3] = (pixelValue shr 16).toByte()  // Canal Rojo
                imageDataOriginal[i * 3 + 1] = (pixelValue shr 8).toByte()  // Canal Verde
                imageDataOriginal[i * 3 + 2] = (pixelValue and 0xFF).toByte()  // Canal Azul
            }

            matImageOriginal.put(0, 0, imageDataOriginal)


            val resizedMat = Mat(width, height, CvType.CV_8UC1)
            Imgproc.resize(mat_mask, resizedMat, Size(width.toDouble(), height.toDouble()))

           // Crear una nueva matriz con tres canales
            val resizedMatRGB = Mat(resizedMat.rows(), resizedMat.cols(), CvType.CV_8UC3)
            Core.merge(List(3) { resizedMat }, resizedMatRGB)
            val gray = Mat()
            Imgproc.cvtColor(resizedMatRGB, gray, Imgproc.COLOR_BGR2GRAY)

            // Crear una MatOfPoint para almacenar los contornos
            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(gray, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            if (contours.isNotEmpty()) {
                val contourSegmented = contours.toList().maxByOrNull { Imgproc.contourArea(it) }

                if(contourSegmented != null){
                    val boundingRect = Imgproc.boundingRect(contourSegmented)


                    val regionRecortada = Mat(matImageOriginal, Rect(boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height))

//                     Convertir la región recortada a un Bitmap
                    val bitmapReturn = Bitmap.createBitmap(regionRecortada.cols(), regionRecortada.rows(), Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(regionRecortada, bitmapReturn)

//                    val random = Random
//                    val randomNumber = random.nextInt()
//                    saveToGallery(bitmapReturn, "placa_$randomNumber")

//                    Log.e("GALERIA  ", " OKK 👌👌👌")
                    return bitmapReturn
                }
                else{
                    Log.e("Contornos ", " No encontrados ")
                }
            }
        return null
    }




    /**
     * Funcion gira la img acuerdo a la rotacion de la camara
     * @param image Bitmap imagen original para aplicar la rotacion
     * @param imageRotation Int angulo de rotacion
     * @return TensorImage?
     */
    private fun processImage(
        image: Bitmap,
        imageRotation: Int
    ): TensorImage? {
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }


    /**
     * Funcion convierte el resultado del tfLite ByteBuffer a un Bitmap
     * @param byteBuffer ByteBuffer
     * @param imgSizeX Int
     * @param imgSizeY Int
     * @return Bitmap?
     */
    private fun convertByteBufferToBitmap(
        byteBuffer: ByteBuffer,
        imgSizeX: Int,
        imgSizeY: Int
    ): Bitmap? {
        byteBuffer.rewind()
        byteBuffer.order(ByteOrder.nativeOrder())
        val bitmap = Bitmap.createBitmap(imgSizeX, imgSizeY, Bitmap.Config.ARGB_4444)
        val pixels = IntArray(imgSizeX * imgSizeY)
        for (i in 0 until imgSizeX * imgSizeY) if (byteBuffer.float == 1f) pixels[i] =
            Color.argb(255, 255, 255, 255) else pixels[i] = Color.argb(0, 0, 0, 0)
        bitmap.setPixels(pixels, 0, imgSizeX, 0, 0, imgSizeX, imgSizeY)
        return bitmap
    }


    /**
     * funcion convierte el bitmap a un Mat de Opencv
     * @param bitmap Bitmap
     * @return Mat
     */
    private fun  bitmaptoMat(
        bitmap: Bitmap
    ): Mat{
        val mat_mask = Mat(256, 256, CvType.CV_8UC1)
        val pixels = IntArray(256 * 256)
        bitmap.getPixels(pixels, 0, 256, 0, 0, 256, 256)
        for (i in 0 until 256) {
            for (j in 0 until 256) {
                val pixelValue = pixels[i * 256 + j]
                val red = Color.red(pixelValue)

                // Crear el valor correspondiente en la matriz
                mat_mask.put(i, j, red.toDouble())
            }
        }
        return mat_mask
    }


    // TODO Función para guardar un Bitmap en la galería
    private fun saveToGallery(bitmap: Bitmap, fileName: String) {
        val directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val file = File(directory, "$fileName.png")

        try {
            val fos = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            fos.flush()
            fos.close()

            // Notificar al sistema que hay nuevos archivos para escanear
            MediaScannerConnection.scanFile(
                context,
                arrayOf(file.toString()),
                null,
                null
            )
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
    //TODO funcion para guardar la mask en un txt
    private  fun SaveTXT ( output: TensorBuffer):  Array<IntArray>{
        Log.e("ConverSALIDA TFLITE", "CONvet init ")
        val shape = output.shape
        val width = shape[1]
        val height = shape[2]
        val binaryMask = Array(width) { IntArray(height) }
        for (i in 0 until width) {
            for (j in 0 until height) {

                // Accede a elementos individuales usando los índices i y j
                val score = output.floatArray[i * height + j]

                binaryMask[i][j] = score.toDouble().roundToInt() * 255

            }
        }

        return binaryMask
    }

    fun resizeBitmap(originalBitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val originalMat = Mat()
        Utils.bitmapToMat(originalBitmap, originalMat)

        val resizedMat = Mat()
        Imgproc.resize(originalMat, resizedMat, Size(targetWidth.toDouble(), targetHeight.toDouble()))

        val resizedBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(resizedMat, resizedBitmap)

        originalMat.release()
        resizedMat.release()

        return resizedBitmap
    }

    private fun escalarImagen(bitmap: Bitmap): Bitmap {

        // Convierte el bitmap a una matriz OpenCV (Mat)
        val imagenMat = Mat()
        Utils.bitmapToMat(bitmap, imagenMat)

        // Convierte la imagen a escala de grises
        Imgproc.cvtColor(imagenMat, imagenMat, Imgproc.COLOR_BGR2GRAY)
        // Aplica el umbral OTSU
        val imagenBinariaOtsu = Mat()
        Imgproc.threshold(imagenMat, imagenBinariaOtsu, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        // Obtén las dimensiones de la imagen binaria OTSU
        val m = imagenBinariaOtsu.rows()
        val n = imagenBinariaOtsu.cols()

        // Realiza la operación de escalado
        val escalada: Mat
        if (m > n) {
            val imgN = Mat.ones(m, ((m - n) / 2).toDouble().roundToInt(), CvType.CV_8UC1)
            Core.multiply(imgN, Scalar(255.0), imgN)
            escalada = Mat.ones(m, m, CvType.CV_8UC1)
            Core.hconcat(Arrays.asList(imgN, imagenBinariaOtsu, imgN), escalada)
        } else {
            val imgN = Mat.ones(((n - m) / 2).toDouble().roundToInt(), n, CvType.CV_8UC1)
            Core.multiply(imgN, Scalar(255.0), imgN)
            escalada = Mat.ones(n, n, CvType.CV_8UC1)
            Core.vconcat(Arrays.asList(imgN, imagenBinariaOtsu, imgN), escalada)
        }

        // Obtén las nuevas dimensiones de la imagen escalada
        val m1 = escalada.rows()
        val n1 = escalada.cols()

        // Añade filas blancas en la parte superior e inferior
        val filasBlancas = Mat.zeros(10, n1, CvType.CV_8UC1)
        filasBlancas.setTo(Scalar(255.0))
        val escalada2 = Mat()
        Core.vconcat(Arrays.asList(filasBlancas, escalada, filasBlancas), escalada2)

        // Redimensiona la imagen a 128x128
        val imagenFinal = Mat()
        Imgproc.resize(escalada2, imagenFinal, Size(128.0, 128.0))

        // Convierte la matriz resultante a un bitmap
        val bitmapResultante = Bitmap.createBitmap(imagenFinal.cols(), imagenFinal.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(imagenFinal, bitmapResultante)

        // Libera memoria
        imagenMat.release()
        imagenBinariaOtsu.release()
        escalada.release()
        filasBlancas.release()
        escalada2.release()
        imagenFinal.release()

        return bitmapResultante
    }

    fun resizeAndConvertToGrayscale(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        // Redimensiona el Bitmap
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)

        // Convierte el Bitmap a escala de grises
        val grayBitmap = Bitmap.createBitmap(resizedBitmap.width, resizedBitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(resizedBitmap, 0f, 0f, paint)

        return grayBitmap
    }


    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * 128 * 128 * 1 * Float.SIZE_BYTES)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(128 * 128)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until 128) {
            for (j in 0 until 128) {
                val valor = intValues[pixel++]
                var normalizedValue = ((valor shr 16) and 0xFF).toFloat()
                normalizedValue = (normalizedValue - 0) / 255.0f
                byteBuffer.putFloat(normalizedValue)

            }
        }
        return byteBuffer
    }

    /**
     * Interfaz encargada de manejar los resultados de esta clase
     */
    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(results: String?, inferenceTime: Long)
        fun onLossResults(lossNumber: Float)
    }


    val modelName =
        when (currentModel) {
            MODEL_RESUNET -> "plateSegmentationML.tflite"
            MODEL_SSDMOBILENET -> "efficientdet-lite0.tflite"
            MODEL_CNN -> "efficientdet-lite1.tflite"
            else -> "plateSegmentationML.tflite"
        }

    companion object {
        const val MODEL_RESUNET = 0
        const val MODEL_SSDMOBILENET = 1
        const val MODEL_CNN = 2


        private const val TAG = "ModelPersonalizationHelper-Mauricio"
    }
}