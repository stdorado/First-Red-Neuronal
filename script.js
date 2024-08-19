//definicion de modelo

const model = tf.sequential()

model.add(tf.layers.dense({
    units : 5, //nodos de la capa oculta
    inputShape : [10], //nodos de capa de entrada
    activation : "relu" // esto devuelve 0 si una entrada es negativa en caso contraria retorna el valor
    //f(x) = max(0,x)
}));

model.add(tf.layers.dense({
    units : 1,
    activation : "sigmoid" // toma la entrada y tranforma el valor entre 0 y 1
    //f(x) = 1 / (1 + exp(-x))
}))

//configuracion del modelo
model.compile({
    optimizer : "adam",//algoritmo que se encarga de ajustar los pesos del modelo AdaGrad + RMSProp
    loss : "binaryCrossentropy", //esta funcion de perdida permite saber que tan bien esta funcionando el modelo
    //midiendo la diferencia entre las prediccion y el valor real
    metrics : ["accuracy"] // esta funcion es la encargada de medir y evaluar la calidad del modelo basado en porcentajes
})

//imprime un resumen de la arquitectura del modelo mostrando cada capa
model.summary()

//genera un tensor de datos aleatorios usando la campana de Gauss
const xs = tf.randomNormal([100,10]) //tamaño del tensor, en este caso 100 ejemplos con 10 caracteristicas cada uno
const ys = tf.randomUniform([100,1],0,1).round() // genera un tensor de datos aleatorios siguiendo una distribucion
//round redondea los valores a 0 o 1


//model entrena al modelo con los datos de entrada xs y las etiquetas ys ajustando los pesos de las capas
// para reducir la perdida y mejorar la precision del modelo
model.fit(xs,ys,{
  epochs : 10, //numero de veces que el modelo vera los datos completos
  callbacks : {
    //imprime la perdida y precision de los datos
    onEpochEnd : (epoch,logs)=>{
        console.log(`Veces que el modelo vio los datos ${epoch + 1}: perdida = ${logs.loss}, precision = ${logs.acc}`);
    }
  }
})