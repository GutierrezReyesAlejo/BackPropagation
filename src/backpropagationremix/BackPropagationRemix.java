package backpropagationremix;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import ia.backpropagation.RedNeuronal;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;


public class BackPropagationRemix {

    /**
     * 1. Se crea una instancia de la RedNeuronal y ésta Red inicializa todo. A
     * partir de ahi, se puede acceder a los metodo de
     *
     * @train( input, target_output ) : Ajusta los pesos de la red.
     * @calculate_error_this_training( input, target_output ) : Calcula el error
     * de la red con los pesos ajustados que tiene
     * @calculate_total_error(set_of_trainings)
     */
//    public static void main(String[] args) {
//        ArrayList<double[]> inputs_training = new ArrayList<>();
//        ArrayList<double[]> outputs_training = new ArrayList<>();
//        try {
//            BufferedReader br = new BufferedReader(new FileReader("/home/alessio/Descargas/iris.data"));            
//            // Leer matriz entrenamiento.            
//            String line = br.readLine();
//            while (line!=null) {
//                String [] tkns = line.split(",");
//                double valsIn[] = new double[tkns.length - 1]; // Omite el ultimo Valor
//                for (int j = 0; j < tkns.length - 1; j++){
//                    valsIn[j] = Double.parseDouble(tkns[j]);
//                }
//                
//                double valsOut[] = new double[3]; // Combinaciones del ultimo Valor
//                switch ( tkns[tkns.length - 1] ){
//                    case "Iris-setosa": valsOut [0] = 1; valsOut [1] = 0; valsOut[2] = 0; break;
//                    case "Iris-versicolor": valsOut [0] = 0; valsOut [1] = 1;  valsOut[2] = 0; break;
//                    case "Iris-virginica": valsOut [0] = 0; valsOut [1] = 0;  valsOut[2] = 1; break;                    
//                }
//                if (valsIn.length == 4 && valsOut.length == 3){
//                    inputs_training.add(valsIn);
//                    outputs_training.add(valsOut);
//                }
//                
//                line = br.readLine();
//            }
//                
//        } catch (FileNotFoundException ex) {
//            System.err.println("Error leyendo el archivo");
//        } catch (IOException ex) {
//            Logger.getLogger(BackPropagationRemix.class.getName()).log(Level.SEVERE, null, ex);
//        }
//        // TODO code application logic here
//        RedNeuronal red = new RedNeuronal(3, new int[]{3,3,3}, 4, 3, 0.8);
//        int n = inputs_training.size();
//        double err = 100.0; int i = 0;
//        do{
//            int x = (int) (Math.random() * 150);
//           red.train( inputs_training.get(x), outputs_training.get(x) );
//           
//            err = red.calculate_total_error(inputs_training, outputs_training);
//            i++;            
//            if (i%5000==0)
//                System.out.format("Error de : %.4f \n", err);
//        }while(i < 100000);
//        
//        err = red.calculate_error_this_training(inputs_training.get(45), outputs_training.get(45));        
//        System.out.format("Epocas=[%d]Error de : %.4f \n", i, err);
//        
//        err = red.calculate_error_this_training(inputs_training.get(90), outputs_training.get(90));        
//        System.out.format("Epocas=[%d]Error de : %.4f \n", i, err);
//        
//        err = red.calculate_error_this_training(inputs_training.get(145), outputs_training.get(145));        
//        System.out.format("Epocas=[%d]Error de : %.4f \n", i, err);
//        
//    }
//    public static void main(String[] args) {
//        RedNeuronal red = new RedNeuronal(2, new int[]{2, 2}, 2, 1, 0.5);
//        ArrayList<double[]> inputs_training = new ArrayList<>();
//        ArrayList<double[]> outputs_training = new ArrayList<>();
//        inputs_training.add(new double[]{0, 0});outputs_training.add(new double[]{0});
//        inputs_training.add(new double[]{0, 1});outputs_training.add(new double[]{1});
//        inputs_training.add(new double[]{1, 0});outputs_training.add(new double[]{1});
//        inputs_training.add(new double[]{1, 1});outputs_training.add(new double[]{0});
//
//        double err = 0.0;
//        int i = 0;
//        do {
//
//            red.train(inputs_training.get(i % 4), outputs_training.get(i % 4));
//
//            err = red.calculate_total_error(inputs_training, outputs_training);
//
//            i++;
//        } while (err > 0.005);
//
//        System.out.format("Error de : %.4f \n", err);
//
//        red.calculate_error_this_training(inputs_training.get(0), outputs_training.get(0));
//        red.calculate_error_this_training(inputs_training.get(1), outputs_training.get(1));
//        red.calculate_error_this_training(inputs_training.get(2), outputs_training.get(2));
//        red.calculate_error_this_training(inputs_training.get(3), outputs_training.get(3));
//
//    }
    public static void main(String[] args) {
        // double rate_train = 1/Math.sqrt(7);
        RedNeuronal red = new RedNeuronal(2, new int[]{13, 13}, 7, 2, 0.01456);
        ArrayList<double[]> inputs_training = new ArrayList<>();
        ArrayList<double[]> outputs_training = new ArrayList<>();
        inputs_training.add(new double[]{142, 0, 20.6, 14.4, 42.8, 46.5, 19.6});
        outputs_training.add(new double[]{0.01, 0.991});
        
        inputs_training.add(new double[]{19, 1, 13.3, 11.1, 27.8, 32.3, 11.3});
        outputs_training.add(new double[]{0.01,0.99});
        
        inputs_training.add(new double[]{169, 0, 16.7, 14.3, 32.3, 37, 14.7});
        outputs_training.add(new double[]{0.99,0.01});
        
        inputs_training.add(new double[]{56, 1, 9.8, 8.9, 20.4, 23.9, 8.8});
        outputs_training.add(new double[]{0.99,0.01});

        double err = 0.0;
        int i = 0;
        do {

            red.train(inputs_training.get(i % 4), outputs_training.get(i % 4));
            err = red.calculate_total_error(inputs_training, outputs_training);
            if (i % 10000 == 0) {
                System.out.format("Error de : %.4f \n", err);
            }
            i++;
        } while (err > 0.15);
 
        ObjectMapper mapper = new ObjectMapper();
        
        File file = new File("artist.json");
        
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
    
    

}
