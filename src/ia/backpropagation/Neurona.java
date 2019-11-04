package ia.backpropagation;

// Subclasses
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.io.Serializable;
import java.util.Random;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Neurona implements Serializable{
    // static variables.
    public static Random random;

    static {
        random = new Random();
    }

    String funcion = "sigmoide";

    public void setFuncion(String funcion) {
        this.funcion = funcion;
    }
    
    // Setter and getters.
    private double inputsX[];
    private double weights[];

    private double output;
    private double valueNet;
    private double errDelta;
    
    // Inicializa los pesos.
    public Neurona(int size_in, double[] X) {
        //random = new Random(System.currentTimeMillis());
        output     = 0.0;
        valueNet    = 0.0;
        errDelta    = 0.0;
        inputsX     = X;
        weights     = new double[size_in];
        
        // TODO. Falta implementar que el usuario ingrese el rango para los numeros aleatorios
        for (int i = 0; i < size_in; i++) {
            //weights[i] = random.nextDouble();  // Funciona para el XOR
             weights[i] = -2 + (random.nextDouble()*4);  // Funciona para Iris.
            //weights[i] = (random.nextDouble()*2);  // Funciona para el ejemplo de PDF
        }
    }
    
    public Neurona(){}

    public Neurona(int szofx) {        
        this(szofx, null);
    }
     
    // E(a) Evaluar la Funcion de Costo o Funcion de Error
    public double calculateError(double target_output) {
        return 0.5 * Math.pow(target_output - this.getOutput(), 2);        
    } 

    public double calculate_output(double[] input) {
        this.inputsX = input;
        double total = 0.0;
        for (int i = 0; i < input.length; i++) {
            total +=weights[i]*input[i];
        }
        this.valueNet = total;
        
        if (funcion.equals("sigmoide")){
            this.output = 1.0 / (1 + Math.pow(Math.E, -valueNet));
        }else if (funcion.equals("lineal")){
            this.output = valueNet;
        }        
        return output;
    }   


    // TODO. This works only to Sigmoidal Neurons. xD
    // δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ = (a - t) * a * (1 - a)
    public double calculateDelta(double target) {
        this.errDelta = (this.getOutput() - target) * this.getOutput() * (1 - this.getOutput());
        return errDelta;
    }
    
    // TODO. This works only to Sigmoidal Neurons too. xD
    // # δ = ∂E/∂zⱼ = dE/dyⱼ * dyⱼ/dzⱼ
    public double calculateDeltaToHiddenNeuron(double total_delta) {
        double delta = total_delta * this.getOutput() * (1 - this.getOutput());
        this.errDelta = delta;
        return delta;
    }
    
    public double ajustarPeso(int i, double rate_training){
        double increase = 0.0;        
        // ∆w = alpha *  delta * Xi
        increase = -rate_training * this.errDelta * this.inputsX[i];
        // w + ∆w
        weights[i] = weights[i] + increase;
        return weights[i];
    }
    
    
    public double getOutput() {
        return output;
    }

    public double getTotalNet() {
        return valueNet;
    }

    public double getErrDelta() {
        return errDelta;
    }   

    public double getWeight(int k) {
        return weights[k];
    }

    public double getInputsX(int k) {
        return inputsX[k];
    }

    int getWeightsLength() {
        return weights.length;
    }

    public static Random getRandom() {
        return random;
    }

    public double[] getInputsX() {
        return inputsX;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getValueNet() {
        return valueNet;
    }    

    public static void setRandom(Random random) {
        Neurona.random = random;
    }

    public void setInputsX(double[] inputsX) {
        this.inputsX = inputsX;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public void setValueNet(double valueNet) {
        this.valueNet = valueNet;
    }

    public void setErrDelta(double errDelta) {
        this.errDelta = errDelta;
    }
}

