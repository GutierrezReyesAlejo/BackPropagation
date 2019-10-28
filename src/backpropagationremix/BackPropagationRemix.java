package backpropagationremix;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import ia.backpropagation.RedNeuronal;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class BackPropagationRemix {

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
    public static void main(String[] args) {
        RedNeuronal red = new RedNeuronal(2, new int[]{2, 2}, 2, 1, 0.5);
        ArrayList<double[]> inputs_training = new ArrayList<>();
        ArrayList<double[]> outputs_training = new ArrayList<>();
        inputs_training.add(new double[]{0, 0}); outputs_training.add(new double[]{0});
        inputs_training.add(new double[]{0, 1}); outputs_training.add(new double[]{1});
        inputs_training.add(new double[]{1, 0}); outputs_training.add(new double[]{1});
        inputs_training.add(new double[]{1, 1}); outputs_training.add(new double[]{0});

        double err = 0.0;
        int i = 0;
        do {

            red.train(inputs_training.get(i % 4), outputs_training.get(i % 4));

            err = red.calculate_total_error(inputs_training, outputs_training);
            if (i%1000==0)
                System.err.println(err);
            i++;
        } while (err > 0.005);

        System.out.format("Error Promedio de : %.4f \n", err);

        // ### Pruebas ### :v
        err = red.calculate_error_this_training(inputs_training.get(0), outputs_training.get(0));
        System.out.format("Error de : %.4f ", err);
        red.pintarOutput();
        
        err = red.calculate_error_this_training(inputs_training.get(1), outputs_training.get(1));
        System.out.format("Error de : %.4f ", err);
        red.pintarOutput();
        
        err = red.calculate_error_this_training(inputs_training.get(2), outputs_training.get(2));
        System.out.format("Error de : %.4f ", err);
        red.pintarOutput();
        
        err = red.calculate_error_this_training(inputs_training.get(3), outputs_training.get(3));
        System.out.format("Error de : %.4f ", err);
        red.pintarOutput();

    }

}
