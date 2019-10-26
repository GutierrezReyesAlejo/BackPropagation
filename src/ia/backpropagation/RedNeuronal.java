package ia.backpropagation;

import ia.backpropagation.Neurona.iFuncionActivacion;
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

    private double[] outputs;

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
        iFuncionActivacion f = buildSigmoideFunction();

        int sizeof_left_layer = size_input;
        for (int i = 0; i < num_hidden_layer; i++) {
            network.add(new Capa(sizeof_left_layer, sizes_of_hidden_layers[i], f));
            sizeof_left_layer = sizes_of_hidden_layers[i];
        }

        outputLayer = new Capa(sizeof_left_layer, size_output, f);
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

    public double calculate_total_error(ArrayList<double[]> training_sets_inputs,
            ArrayList<double[]> training_sets_outputs) {
        try {
            Guardia.againstDifferentSize(size_input, size_output, training_sets_inputs, training_sets_outputs);

            double total_error = 0.0;
            for (int i = 0; i < training_sets_inputs.size(); i++) {
                double input[] = training_sets_inputs.get(i);
                double out_target[] = training_sets_outputs.get(i);

                this.feed_forward(input);

                int for_this_neuron = 0;
                for (Neurona neuron : outputLayer.getNeuronas()) {
                    total_error += neuron.calculateError(out_target[for_this_neuron++]);
                }
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
            
            System.out.print("Input: ");
            for (int i = 0; i < input.length; i++) {
                System.out.format("[%.2f]", input[i]);
                
            }
            
            System.out.println("\tOutput: ");
            for (int i = 0; i < outputs.length; i++) {
                System.out.format("[%.4f]vs[%.4f] \n", outputs[i], target_output[i]);
                
            }
            
            return total_error;
        } catch (Exception ex) {
            Logger.getLogger(RedNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }
        return 0;
    }
    
    
    public void exportRedNeuronalToTxt(String name_file){
        
    }
    
    public static RedNeuronal loadRedFromTxtFile(String txt_file){
        
        return null;
    }

    public iFuncionActivacion buildSigmoideFunction() {
        iFuncionActivacion f = new iFuncionActivacion() {
            @Override
            public double evalFunction(double net) {
                return 1.0 / (1 + Math.pow(Math.E, -net));
            }

            // dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
            // The derivative (not partial derivative since there is only one variable) of the output then is:
            @Override
            public double evalPartialDerivate(double output) {
                return output * (1 - output);
            }
        };
        return f;
    }

    public iFuncionActivacion buildLinealFunction() {
        iFuncionActivacion f = new iFuncionActivacion() {
            @Override
            public double evalFunction(double net) {
                return net;
            }

            @Override
            public double evalPartialDerivate(double net) {
                return 1.0;
            }
        };
        return f;
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

    
}
