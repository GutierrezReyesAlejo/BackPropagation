package ia.backpropagation;

import java.util.ArrayList;

public class Guardia {

    public static void againstNullPointer(Object o, String Objectalias) {
        if (o == null) {
            throw new NullPointerException(Objectalias + " Has a Null Value. Program Can't continue from this point. Please");
        }
    }

    public static void againstUnexpectedValue(Object get, Object tarjet) {
        if (get instanceof Integer || get instanceof Double
                || get instanceof Float) {

        }
    }

    public static void againstZeroValues(double given, String VarAlias) throws Exception {
        if (given <= 0) {
            throw new Exception("Var [" + VarAlias + "]" + " Has a Not Expected Value that is : " + given);
        }
    }

    public static void againstDifferentSize(int sz_input, int out_size, ArrayList<double[]> train_list, ArrayList<double[]> outs_tarjet) throws Exception {
        if (train_list.size() != outs_tarjet.size()) {
            throw new Exception("Las listas deben ser del mismo tamaño");
        }
        
        for (int i = 0; i < outs_tarjet.size(); i++) {
            if ( train_list.get(i).length!=sz_input || outs_tarjet.get(i).length != out_size )
                throw new Exception(String.format("[%d]Los tamaños adecuados para las Entradas y Salidas son [%d][%d] vs [%d][%d]", i,
                        sz_input, out_size, train_list.get(i).length, outs_tarjet.get(i).length ));
        }
    }
    
    public static void againstDifferentSize(int sz_input, int out_size, double[] train_list, double[] outs_tarjet) throws Exception {
        if ( train_list.length!=sz_input || outs_tarjet.length != out_size )
                throw new Exception(String.format("Los tamaños adecuados para las Entradas y Salidas son [%d][%d]", sz_input, out_size));
    }

    static void againstDifferentSize(double[] x, double[] inputsW) throws Exception {
        if (x.length!= inputsW.length) {
            throw new Exception("Las listas deben ser del mismo tamaño");
        }
    }

    static void againstNotNxMArray(ArrayList<double[]> list) throws Exception {
        int const_size = list.get(0).length;
        for (int i = 1; i < list.size(); i++) {
            if (list.get(i).length != const_size){
                throw new Exception("La lista no es de tamaño proporcional M x N : [][]");
            }
            
        }
    }

    static void againstDifferentSize(double[] input, int size_input) throws Exception {
        if (input.length != size_input)throw new Exception("El tamaño esperado para el input es de : " + size_input + " vs " + input.length);
    }
}
