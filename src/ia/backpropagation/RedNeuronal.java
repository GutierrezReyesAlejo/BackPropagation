package ia.backpropagation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RedNeuronal implements Serializable{

    // Setters and Getter 
    private int size_input;
    private int size_output;
    private double rate_training = 0.9;
    private ArrayList<Capa> network;
    private Capa outputLayer;
    int cont=0;

    private double[] outputs;

    public RedNeuronal(){}
    /**
     * Crea una red neuronal. Un conjunto de capas ocultas y una capa de salida
     * {L1, L2, L3,.. Lout} con tamaños {N1, N2, N3... Nout} y funciones de
     * activacion {F1, F2, F3...Fout} para cada capa. Por defecto usaremos la
     * Sigmoide
     */
    public RedNeuronal(int num_hidden_layer,
            int[] sizes_of_hidden_layers,
            int size_input, int size_output, double rate_training) {

        this.size_input = size_input;
        this.size_output = size_output;
        this.rate_training = rate_training;

        network = new ArrayList<>();

        int sizeof_left_layer = size_input;
        for (int i = 0; i < num_hidden_layer; i++) {
            network.add(new Capa(sizeof_left_layer, sizes_of_hidden_layers[i]));
            sizeof_left_layer = sizes_of_hidden_layers[i];
        }

        outputLayer = new Capa(sizeof_left_layer, size_output);
    }

    /**
     * 1. Alimentar Red 2. Calcular Deltas o Errores. 3. Ajustar pesos de la Red
     * 4.
     */
    public void train(double[] inputs, double[] training_output) {
        try {
            Guardia.againstDifferentSize(size_input, size_output, inputs, training_output );
            
            this.feed_forward(inputs);
            // # Neuron Deltas
            // Output Neuron Delta
            for (int i = 0; i < outputLayer.getSize(); i++) {
                outputLayer.getNeuronas().get(i).calculateDelta(training_output[i]);
            }
            // Hidden Neuron Delta. Propagar el Error.
            Capa layer_forward = outputLayer;
            for (int i = network.size() - 1; i >= 0; i--) {
                Capa hiddenLayer = network.get(i);
                
                for (int indexOfThisNeuron = 0; indexOfThisNeuron < hiddenLayer.getNeuronas().size(); indexOfThisNeuron++) {
                    
                    // # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                    Neurona hiddenNeuron = hiddenLayer.getNeurona(indexOfThisNeuron);
                    double total_dE_dy = 0.0;
                    for (int k = 0; k < layer_forward.getSize(); k++) {
                        total_dE_dy +=          layer_forward.getNeurona(k).
                                getErrDelta() * layer_forward.getNeurona(k).getWeight(indexOfThisNeuron);
                    }
                    // # ∂E/∂zⱼ = dE/dyⱼ * dyⱼ/dzⱼ
                    hiddenNeuron.calculateDeltaToHiddenNeuron(total_dE_dy);
                }
                
                layer_forward = network.get(i);
            }
            
            // # Update Weights
            for (int i = 0; i < outputLayer.getSize(); i++) {
                for (int j = 0; j < outputLayer.getNeurona(i).getWeightsLength(); j++) {
                    outputLayer.getNeurona(i).ajustarPeso(j, this.rate_training);
                }
            }
            
            for (int i = 0; i < network.size(); i++) {  // Capa Oculta
                Capa hiddenLayer = network.get(i);
                for (int indexOfThisNeuron = 0; indexOfThisNeuron < hiddenLayer.getNeuronas().size(); indexOfThisNeuron++) { // Neurona
                    Neurona hiddenNeuron = hiddenLayer.getNeurona(indexOfThisNeuron);
                    for (int peso = 0; peso < hiddenNeuron.getWeightsLength(); peso++) {
                        hiddenNeuron.ajustarPeso(peso, this.rate_training);
                    }
                }
            }
        } catch (Exception ex) {
            System.err.println("Sorry Crack ;-), Atiende el error: " + ex.getMessage());
        }

    }

    // Alimenta la red y vacia la salida en el vector de salida de este objeto.
    public void feed_forward(double[] input) {
        try {
            Guardia.againstNullPointer(this.outputLayer, "Capa de Salida");
            Guardia.againstDifferentSize(input, size_input);

            double[] ouputs_from_hidden_layer = null;

            Capa firstHiddenLayer = network.get(0);
            ouputs_from_hidden_layer = firstHiddenLayer.feed_forward(input);

            for (int i = 1; i < network.size(); i++) {
                Capa hiddeLayer = network.get(i);
                ouputs_from_hidden_layer = hiddeLayer.feed_forward(ouputs_from_hidden_layer);
            }

            this.outputs = outputLayer.feed_forward(ouputs_from_hidden_layer);
        } catch (Exception ex) {
            Logger.getLogger(RedNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public double calculate_total_error(
           
            ArrayList<double[]> training_sets_inputs,
            ArrayList<double[]> training_sets_outputs) {
        
        try {
            cont=0;
            Guardia.againstDifferentSize(size_input, size_output, training_sets_inputs, training_sets_outputs);

            double total_error = 0.0;
            
            for (int i = 0; i < training_sets_inputs.size(); i++) {
                double input[] = training_sets_inputs.get(i);
                double out_target[] = training_sets_outputs.get(i);

                this.feed_forward(input);

                int for_this_neuron = 0;
                double error_patron=0.0;
                for (Neurona neuron : outputLayer.getNeuronas()) {
                    error_patron += neuron.calculateError(out_target[for_this_neuron++]);
                }
                if(error_patron<0.13)
                    cont++;
                
                total_error+=error_patron;
            }
            
            return total_error/training_sets_inputs.size();
        } catch (Exception ex) {
            Logger.getLogger(RedNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }
        return 0;
    }

    public double calculate_error_this_training(double input[], double[] target_output) {
        try {
            Guardia.againstDifferentSize(size_input, size_output, input, target_output);
            this.feed_forward(input);
            double total_error = 0.0;
            
            int for_this_neuron = 0;
            for (Neurona neuron : outputLayer.getNeuronas()) {
                total_error += neuron.calculateError(target_output[for_this_neuron++]);
            }
            return total_error;
        } catch (Exception ex) {
            Logger.getLogger(RedNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }
        return 0;
    }
    
    public int getSize_input() {
        return size_input;
    }

    public int getSize_output() {
        return size_output;
    }

    public double getRate_training() {
        return rate_training;
    }

    public ArrayList<Capa> getNetwork() {
        return network;
    }

    public double[] getOutputs() {
        return outputs;
    }

    public Capa getOutputLayer() {
        return outputLayer;
    }   
    

    public void setSize_input(int size_input) {
        this.size_input = size_input;
    }

    public void setSize_output(int size_output) {
        this.size_output = size_output;
    }

    public void setRate_training(double rate_training) {
        this.rate_training = rate_training;
    }

    public void setNetwork(ArrayList<Capa> network) {
        this.network = network;
    }

    public void setOutputLayer(Capa outputLayer) {
        this.outputLayer = outputLayer;
    }

    public void setOutputs(double[] outputs) {
        this.outputs = outputs;
    }    

    public void pintarOutput() {
        int round_value = 0;
        for (int i = 0; i < outputs.length; i++) {            
            round_value = outputs[i]<0.5?0:1;
            
            System.out.format("[%d]", round_value);
        }
        System.out.println();
    }

    public int getCont() {
        return cont;
    }
    
    
}
