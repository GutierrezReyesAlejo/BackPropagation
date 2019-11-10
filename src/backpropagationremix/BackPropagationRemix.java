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
    static int ancho = 47;
    static int alto = 69;

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
        si.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        si.setVisible(true);
        
        Random random = new Random();
        
        ArrayList<String> inputs = new ArrayList<>();
        ArrayList<double[]> outputs = new ArrayList<>();
        
        
        try {
            BufferedReader br = new BufferedReader(new FileReader("numeros.txt"));
            String line = br.readLine();
            
            while (line != null) {
                String[] tkns = line.split(",");
                String valsIn=""; // Omite el ultimo Valor
                for (int j = 0; j < tkns.length - 1; j++) {
                    valsIn = tkns[j];
                }

                double valsOut[] = new double[10]; // Combinaciones del ultimo Valor
                for (int i = 0; i < valsOut.length; i++) {
                    valsOut[i]=0; 
                }
                switch (tkns[tkns.length - 1]) {
                    case "0":
                        valsOut[0] = 1;
                        break;
                    case "1":
                        valsOut[1] = 1;
                        break;
                    case "2":
                        valsOut[2] = 1;
                    break;
                    case "3":
                        valsOut[3] = 1;
                    break;
                    case "4":
                        valsOut[4] = 1;
                    break;
                    case "5":
                        valsOut[5] = 1;
                    break;
                    case "6":
                        valsOut[6] = 1;
                    break;
                    case "7":
                        valsOut[7] = 1;
                    break;
                    case "8":
                        valsOut[8] = 1;
                    break;
                    case "9":
                        valsOut[9] = 1;
                    break;
                }
                
                if ( valsOut.length == 10) {
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
        
        int mitad = 100  ;   //Numero de vectores de prueba
        int pos;
        
        while(pos_pruebas.size()<mitad){    //Llenar arrglo deentero con las posiciones aleatorias
            pos=random.nextInt(inputs.size());
            if(!seRepite(pos))
                pos_pruebas.add(pos);
        }
        
        
        ArrayList<double[]> inputs_training = new ArrayList<>();
        ArrayList<double[]> outputs_training = new ArrayList<>();
        
        ArrayList<double[]> inputs_prueba = new ArrayList<>();
        ArrayList<double[]> outputs_prueba = new ArrayList<>();
        
        for(int i=0; i<inputs.size(); i++){
            if(pos_pruebas.indexOf(i)==-1){
                inputs_training.add(obtenerArray(invertir(obtenerImagen(inputs.get(i)))));
                outputs_training.add(outputs.get(i));
            }  
            else{
                inputs_prueba.add(obtenerArray(invertir(obtenerImagen(inputs.get(i)))));
                outputs_prueba.add(outputs.get(i));
            }           
        }

        System.err.println("Ya acabe");
        
        /////////////
        
        
        // TODO Creando la Red y 	  No. Capas ocultas | Tamaños de las capas Ocultas                   |nInputs | nOutputs | Tasa aprendizaje
        int tam=ancho*alto;
        RedNeuronal red = new RedNeuronal(1, new int[]{1000}, tam, 10,  0.06);
        int n = inputs_training.size();
        double err = 100.0;
       // int i = 0;
        System.err.println(inputs_training.size());
        do {
           // int x = (int) (Math.random() * n); // Escoge un numero al azar
            for(int j=0; j<inputs_training.size(); j++){
                //double p[] = obtenerArray(   obtenerImagen(inputs_training.get(j)));
                red.train(inputs_training.get(j), outputs_training.get(j));   // Entrena a la red con ese Patron.
            }

            err = red.calculate_total_error(inputs_training, outputs_training);  // Calcula el error promedio Respecto a todos los Patrones.
         //   i++;
          //  if (i % 100 == 0) {
             System.out.format("Error de : %.4f \n", err); // Monitoreando Ando... :v
           // }
        } while (err > 0.01 && !parar);
        System.err.println("\n"+err);
        System.err.println(red.getCont());

        err = red.calculate_total_error(inputs_prueba, outputs_prueba);
        System.err.println("\n\n"+err);
        System.err.println(red.getCont());
        serializar("ImagenesChidasTodas0y1_11.json", red);
        
        System.err.println("===================================================\n");
        double c[]=obtenerArray(invertir(obtenerImagen("numeros/0/0_105.png")));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(invertir(obtenerImagen("numeros/0/0_50.png")));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(invertir(obtenerImagen("numeros/1/1_41.png")));
        red.feed_forward(c);
        red.pintarOutput();
        System.err.println("===================================================\n");
        c=obtenerArray(invertir(obtenerImagen("numeros/1/1_91.png")));
        red.feed_forward(c);
        red.pintarOutput();
        si.dispose();
    }

 
/*
    public static void main(String[] args) throws IOException {
        System.load("C:/opencv/build/java/x64/opencv_java411.dll");
        RedNeuronal red = deserializar("ImagenesChidasTodas0y1_11.json");

        for (int i = 1; i < 17; i++) {
            System.err.println("===================================================\n");
            double c[]=obtenerArray(obtenerImagen("numeros/1/1/"+i+".png"));
            red.feed_forward(c);
            red.pintarOutput();
        }
          
    } 
*/

    public static boolean seRepite(int pos){
        for(int e : pos_pruebas)
           if(e==pos)
               return true;
        return false;
    }
     
    
    
    
    public static double[] obtenerArray(Mat imagen){
        double array[] = new double[imagen.width()*imagen.height()];
        int cont=0;
        for (int i = 0; i < imagen.height(); i++) {
            for (int j = 0; j < imagen.width(); j++) {
               if(imagen.get(i, j)[0]==0)
                   array[cont]=0;
               else
                   array[cont]=1;
               
               cont++;
            }
        }   
        return array;
    }
    
    public static Mat obtenerImagen(String ruta){
        Mat imagen = Imgcodecs.imread(ruta, Imgcodecs.IMREAD_ANYCOLOR);
        Mat imagenredim = new Mat();
        Size sz = new Size(ancho,alto);
        Imgproc.resize( imagen, imagenredim, sz );
        
        
        Mat gris=new Mat(imagenredim.width(),imagenredim.height(),imagenredim.type());       
        Imgproc.cvtColor(imagenredim, gris, Imgproc.COLOR_RGB2GRAY);

        Mat binario = new Mat(gris.width(),gris.height(),gris.type());
        Imgproc.threshold(gris, binario, 70, 255, Imgproc.THRESH_OTSU);
        
        return binario;
    }
    
    public static boolean cambiar(double arre[]){
        int bl=0;
        int ng=0;
        
        for (double d : arre) {
            if(d==0) ng++;
            else     bl++;
        }
 
        if(bl>ng)
            return true;
        return false;
    }
    
    public static Mat invertir(Mat imagen){
        double arre[] = obtenerArray(imagen);
        if(cambiar(arre))
            Imgproc.threshold(imagen, imagen, 70, 255, Imgproc.THRESH_BINARY_INV);
        return imagen;
    }

}
