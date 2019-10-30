package backpropagationremix;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import ia.backpropagation.RedNeuronal;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JButton;
import javax.swing.JFrame;

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
    
    public static void main(String[] args) {
        
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
        
        ArrayList<double[]> inputs = new ArrayList<>();
        ArrayList<double[]> outputs = new ArrayList<>();
        
        try {
            BufferedReader br = new BufferedReader(new FileReader("iris.data"));
            // Leer matriz entrenamiento.
           // System.out.println(br.lines().count());
            String line = br.readLine();
            
            while (line != null) {
                String[] tkns = line.split(",");
                double valsIn[] = new double[tkns.length - 1]; // Omite el ultimo Valor
                for (int j = 0; j < tkns.length - 1; j++) {
                    valsIn[j] = Double.parseDouble(tkns[j]);
                }

                double valsOut[] = new double[3]; // Combinaciones del ultimo Valor
                switch (tkns[tkns.length - 1]) {
                    case "Iris-setosa":
                        valsOut[0] = 1;
                        valsOut[1] = 0;
                        valsOut[2] = 0;
                        break;
                    case "Iris-versicolor":
                        valsOut[0] = 0;
                        valsOut[1] = 1;
                        valsOut[2] = 0;
                        break;
                    case "Iris-virginica":
                        valsOut[0] = 0;
                        valsOut[1] = 0;
                        valsOut[2] = 1;
                        break;
                }
                if ( valsOut.length == 3) {
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
        
        
        ArrayList<double[]> inputs_training = new ArrayList<>();
        ArrayList<double[]> outputs_training = new ArrayList<>();
        
        ArrayList<double[]> inputs_prueba = new ArrayList<>();
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
        
        /////////////
        
        
        // TODO Creando la Red y 	  No. Capas ocultas | Tamaños de las capas Ocultas                   |nInputs | nOutputs | Tasa aprendizaje
        RedNeuronal red = new RedNeuronal(3, new int[]{8,4,6}, 4, 3,  0.06);
        int n = inputs_training.size();
        double err = 100.0;
        int i = 0;
        do {
            int x = (int) (Math.random() * n); // Escoge un numero al azar
            red.train(inputs_training.get(x), outputs_training.get(x));   // Entrena a la red con ese Patron.

            err = red.calculate_total_error(inputs_training, outputs_training);  // Calcula el error promedio Respecto a todos los Patrones.
            i++;
            if (i % 10000 == 0) {
                System.out.format("Error de : %.4f \n", err); // Monitoreando Ando... :v
            }
        } while (err > 0.01 && !parar);
        System.err.println("\n"+err);
        System.err.println(red.getCont());

        err = red.calculate_total_error(inputs_prueba, outputs_prueba);
        System.err.println("\n\n"+err);
        System.err.println(red.getCont());
        //serializar("Iris-example3.json", red);
        
        System.err.println("===================================================\n");
        red.feed_forward(new double[]{4.9,3.0,1.4,0.2});
        red.pintarOutput();
        si.dispose();
    }
    
    
    
    public static boolean seRepite(int pos){
        for(int e : pos_pruebas)
           if(e==pos)
               return true;
        return false;
    }

}
