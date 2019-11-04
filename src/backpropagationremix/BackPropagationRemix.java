package backpropagationremix;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.sun.org.apache.xerces.internal.impl.dv.util.Base64;
import ia.backpropagation.RedNeuronal;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JFrame;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class BackPropagationRemix {
    static int ancho = 94;
    static int alto = 138;

    public static RedNeuronal deserializar(String f) {
        ObjectMapper mapper = new ObjectMapper();
        File file = new File(f);
        RedNeuronal red = null;
        try {
            file.createNewFile();
            // mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
            // Serialize Java object info JSON file.
            red = mapper.readValue(file, RedNeuronal.class);

        } catch (IOException e) {
            e.printStackTrace();
        }
        return red;

    }

    public static void serializar(String f, RedNeuronal red) {
        ObjectMapper mapper = new ObjectMapper();

        File file = new File(f);

        try {
            file.createNewFile();
            mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
            // Serialize Java object info JSON file.
            mapper.writeValue(file, red);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(":)");
    }
    
    // ### Iris.data Example.

//    public static void main(String[] args) {
//        RedNeuronal red = deserializar("Iris-example.json");
//        
//        // # 7.7,3.8,6.7,2.2,Iris-virginica
//        double err = red.calculate_error_this_training(new double[]{6.7,3.3,5.7,2.1}, new double[]{0.0,0.0,1.0});
//        System.out.format("Error: %.3f \t", err);
//        red.pintarOutput();
//        // # 5.8,2.7,4.1,1.0,Iris-versicolor
//        err = red.calculate_error_this_training(new double[]{5.8,2.7,4.1,1.0}, new double[]{0.0,1.0,0.0});
//        System.out.format("Error: %.3f \t", err);
//        red.pintarOutput();
//        // # 5.7,4.4,1.5,0.4,Iris-setosa
//        err = red.calculate_error_this_training(new double[]{5.7,4.4,1.5,0.4}, new double[]{1.0,0.0,0.0});
//        System.out.format("Error: %.3f \t", err);
//        red.pintarOutput();
//    }
    
    
    
    // # XOR example
//    public static void main(String[] args) {
//        RedNeuronal red = deserializar("ejemplo-pdf.json");
//
//        ArrayList<double[]> inputs_training = new ArrayList<>();
//        ArrayList<double[]> outputs_training = new ArrayList<>();
//        inputs_training.add(new double[]{142, 0, 20.6, 14.4, 42.8, 46.5, 19.6});
//        outputs_training.add(new double[]{0.0, 1});
//
//        inputs_training.add(new double[]{19, 1, 13.3, 11.1, 27.8, 32.3, 11.3});
//        outputs_training.add(new double[]{0.0, 1});
//
//        inputs_training.add(new double[]{169, 0, 16.7, 14.3, 32.3, 37, 14.7});
//        outputs_training.add(new double[]{1, 0.0});
//
//        inputs_training.add(new double[]{56, 1, 9.8, 8.9, 20.4, 23.9, 8.8});
//        outputs_training.add(new double[]{1, 0.0});
//
//        double err = red.calculate_error_this_training(inputs_training.get(0), outputs_training.get(0));
//        System.err.println(err);
//        err = red.calculate_total_error(inputs_training, outputs_training);
//        System.err.println(err);
//        
//        red.feed_forward(new double[]{56, 1, 9.8, 8.9, 20.4, 23.9, 8.8});
//        red.pintarOutput();
//        
//        
//       
//
//    }
    
    static ArrayList<Integer> pos_pruebas = new ArrayList<>();
    static boolean parar=false;
    
    static void para(){
        parar=true;
    }
    
    public static void main(String[] args) throws IOException {
        System.load("C:/opencv/build/java/x64/opencv_java411.dll");
        
        
        JFrame si = new JFrame();
        JButton boton = new JButton("Parar");
        boton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent ae) {
                para();
            }
        });
        si.add(boton);
        si.setSize(100, 100);
        si.setVisible(true);
        
        Random random = new Random();
        
        ArrayList<String> inputs = new ArrayList<>();
        ArrayList<double[]> outputs = new ArrayList<>();
        
        try {
            BufferedReader br = new BufferedReader(new FileReader("imagenes.txt"));
            // Leer matriz entrenamiento.
           // System.out.println(br.lines().count());
            String line = br.readLine();
            
            while (line != null) {
                String[] tkns = line.split(",");
                String valsIn=""; // Omite el ultimo Valor
                for (int j = 0; j < tkns.length - 1; j++) {
                    valsIn = tkns[j];
                }

                double valsOut[] = new double[2]; // Combinaciones del ultimo Valor
                switch (tkns[tkns.length - 1]) {
                    case "0":
                        valsOut[0] = 1;
                        valsOut[1] = 0;
                        break;
                    case "1":
                        valsOut[0] = 0;
                        valsOut[1] = 1;
                        break;
                }
                if ( valsOut.length == 2) {
                    inputs.add(valsIn);
                    outputs.add(valsOut);
                }

                line = br.readLine();
            }

        } catch (FileNotFoundException ex) {
            System.err.println("Error leyendo el archivo");
        } catch (IOException ex) {
            Logger.getLogger(BackPropagationRemix.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        //Llenar arreglos a usar        
        
        int mitad = inputs.size()/2;   //Numero de vectores de prueba
        int pos;
        
        while(pos_pruebas.size()<mitad){    //Llenar arrglo deentero con las posiciones aleatorias
            pos=random.nextInt(inputs.size());
            if(!seRepite(pos))
                pos_pruebas.add(pos);
        }
        
        
        ArrayList<String> inputs_training = new ArrayList<>();
        ArrayList<double[]> outputs_training = new ArrayList<>();
        
        ArrayList<String> inputs_prueba = new ArrayList<>();
        ArrayList<double[]> outputs_prueba = new ArrayList<>();
        
        for(int i=0; i<inputs.size(); i++){
            if(pos_pruebas.indexOf(i)==-1){
                inputs_training.add(inputs.get(i));
                outputs_training.add(outputs.get(i));
            }  
            else{
                inputs_prueba.add(inputs.get(i));
                outputs_prueba.add(outputs.get(i));
            }           
        }
        
        ArrayList<double[]> inputs_training2 = new ArrayList<>();
        ArrayList<double[]> outputs_training2 = new ArrayList<>();
        
        ArrayList<double[]> inputs_prueba2 = new ArrayList<>();
        ArrayList<double[]> outputs_prueba2 = new ArrayList<>();
        
        for(int i=0; i<inputs_training.size(); i++){
            inputs_training2.add(obtenerArray(obtenerImagen(inputs_training.get(i))));
            outputs_training2.add(outputs_training.get(i));
            
            inputs_prueba2.add(obtenerArray(obtenerImagen(inputs_prueba.get(i))));
            outputs_prueba2.add(outputs_prueba.get(i));
        }

        
        /////////////
        
        
        // TODO Creando la Red y 	  No. Capas ocultas | Tamaños de las capas Ocultas                   |nInputs | nOutputs | Tasa aprendizaje
        int tam=ancho*alto;
        RedNeuronal red = new RedNeuronal(5, new int[]{350,300,100,50,50}, tam, 2,  0.05);
        //int n = inputs_training.size();
        double err = 100.0;
        int i = 0;
        
        do {
            //int x = (int) (Math.random() * n); // Escoge un numero al azar
            for(int j=0; j<inputs_training2.size(); j++){
                //double p[] = obtenerArray(   obtenerImagen(inputs_training.get(j)));
                red.train(inputs_training2.get(j), outputs_training2.get(j));   // Entrena a la red con ese Patron.
            }

            err = red.calculate_total_error(inputs_training2, outputs_training2);  // Calcula el error promedio Respecto a todos los Patrones.
         //   i++;
          //  if (i % 100 == 0) {
                System.out.format("Error de : %.4f \n", err); // Monitoreando Ando... :v
           // }
        } while (err > 0.01 && !parar);
        System.err.println("\n"+err);
        System.err.println(red.getCont());

        err = red.calculate_total_error(inputs_prueba2, outputs_prueba);
        System.err.println("\n\n"+err);
        System.err.println(red.getCont());
        serializar("ImagenesChidas0y1_3.json", red);
        
        System.err.println("===================================================\n");
        double c[]=obtenerArray(obtenerImagen("numeros/0/0_105.png"));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(obtenerImagen("numeros/0/0_50.png"));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(obtenerImagen("numeros/1/1_41.png"));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(obtenerImagen("numeros/1/1_91.png"));
        red.feed_forward(c);
        red.pintarOutput();
        si.dispose();
    }
    
 /*
    public static void main(String[] args) throws IOException {
        System.load("C:/opencv/build/java/x64/opencv_java411.dll");
        RedNeuronal red = deserializar("ImagenesChidas0y1_1.json");
        
         System.err.println("===================================================\n");
        double c[]=obtenerArray(obtenerImagen("numeros/1/1_75.png"));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(obtenerImagen("numeros/1/1_76.png"));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(obtenerImagen("numeros/1/1_77.png"));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(obtenerImagen("numeros/1/1_78.png"));
        red.feed_forward(c);
        red.pintarOutput();
        
    }
*/    
    public static boolean seRepite(int pos){
        for(int e : pos_pruebas)
           if(e==pos)
               return true;
        return false;
    }
     
    
    
    
    public static double[] obtenerArray(Mat imagen){
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".png", imagen, matOfByte); 
 
        byte[] byteArray = matOfByte.toArray();
        BufferedImage bufImage = null;
        
        try {
            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }

        final byte[] pixels = ((DataBufferByte) bufImage.getRaster().getDataBuffer()).getData();
        final int width = bufImage.getWidth();
        final int height = bufImage.getHeight();

        double array[] = new double[height*width];
        
        final int pixelLength = 3;
        int cont=0;
        for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
            int argb = 0;
            //argb += -16777216; // 255 alpha
            argb += ((int) pixels[pixel] & 0xff); // blue
            argb += (((int) pixels[pixel + 1] & 0xff) << 8); // green
            argb += (((int) pixels[pixel + 2] & 0xff) << 16); // red
            if(argb==0)
                array[cont]=0;
            else
                array[cont]=1;
            cont++;

            col++;
            if (col == width) {
               col = 0;
               row++;
            }
        }
        
        return array;
    }
    
    public static Mat obtenerImagen(String ruta){
        Mat imagen = Imgcodecs.imread(ruta, Imgcodecs.IMREAD_ANYCOLOR);
        
        Mat gris=new Mat(imagen.width(),imagen.height(),imagen.type());
        Mat blur=new Mat(imagen.width(),imagen.height(),imagen.type());
        Mat canny = new Mat(imagen.width(),imagen.height(),imagen.type());
        
        Imgproc.cvtColor(imagen, gris, Imgproc.COLOR_RGB2GRAY);
        Size s = new Size(3,3);
        int min_threshold=50;
        int ratio = 3;
        Imgproc.blur(gris, blur,s);
        Imgproc.Canny(blur, canny, min_threshold,min_threshold*ratio);
        Mat imagenredim = new Mat();
        Size sz = new Size(ancho,alto); //dependera de las imagenes del dataset
        Imgproc.resize( canny, imagenredim, sz );
        Mat binario = new Mat(gris.width(),gris.height(),gris.type());
        Imgproc.threshold(imagenredim, binario, 100, 255, Imgproc.THRESH_BINARY);
        
        return binario;
    }

}
