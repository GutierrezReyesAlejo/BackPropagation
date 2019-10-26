package ia.backpropagation;

// Subclasses
import java.io.Serializable;
import java.util.Random;

public class Neurona implements Serializable{
    // static variables.
    public static Random random;

    static {
        random = new Random();
    }

    // Setter and getters.
    private double inputsX[];
    private double weights[];

    private double output;
    private double valueNet;
    private double errDelta;
    
    private iFuncionActivacion functionA;

    // Inicializa los pesos.
    public Neurona(int size_in, double[] X, iFuncionActivacion f) {
        output     = 0.0;
        valueNet    = 0.0;
        errDelta         = 0.0;
        functionA   = f;
        inputsX     = X;
        weights     = new double[size_in];
        
        
        for (int i = 0; i < size_in; i++) {
            // weights[i] = -2 + (random.nextDouble()*4);
            weights[i] = -2 + (random.nextDouble()*3.56);
        }
    }

    public Neurona(int szofx, iFuncionActivacion f) {        
        this(szofx, null, f);
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
        this.output = functionA.evalFunction(total);
        return output;
    }   


    // δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ = (a - t) * a * (1 - a)
    public double calculateDelta(double target) {
        this.errDelta = (this.getOutput() - target) * this.getOutput() * (1 - this.getOutput());
        return errDelta;
    }
    
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

    public iFuncionActivacion getFunctionA() {
        return functionA;
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
    
    
    
    /**
     * 
     * 
     * 
     *      INTERFACES
     * 
     * 
     * 
     */
    public interface iFuncionActivacion extends Serializable{

        double evalFunction(double net);

        double evalPartialDerivate(double net);
    }
}

