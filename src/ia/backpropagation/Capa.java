package ia.backpropagation;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.io.Serializable;
import java.util.ArrayList;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Capa implements Serializable{
    // Getter and Setters.
    ArrayList<Neurona> neuronas;

    
    public Capa(){}
    
    public Capa(int sizeof_left_layer, int num_neurons) {
        neuronas = new ArrayList<>();
        for (int i = 0; i < num_neurons; i++) {
            neuronas.add(new Neurona(sizeof_left_layer));
        }
    }

    // Compute their Neurons and get their Activations Outputs.        
    public double[] getOutput() {
        double output[] = new double[this.neuronas.size()];
        int i = 0;
        for (Neurona n : this.neuronas) {
            output[i++] = n.getOutput();
        }
        return output;
    }

    public int getSize() {
        return this.neuronas.size();
    }
    
    public ArrayList<Neurona> getNeuronas(){
        return neuronas;
    }
    
    public Neurona getNeurona(int index){
        return neuronas.get(index);
    }
    

    public double[] feed_forward(double[] input) {
        double output[] = new double[this.neuronas.size()];
        int i = 0;
        for (Neurona n : this.neuronas) {
            output[i++] = n.calculate_output(input);
        }
        return output;
    }

    public void setNeuronas(ArrayList<Neurona> neuronas) {
        this.neuronas = neuronas;
    }
}
