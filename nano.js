class NanoNN {
  constructor() {
    // Set up fast math buffer for bit-level optimizations
    this._f32Buffer = new ArrayBuffer(4);
    this._f32 = new Float32Array(this._f32Buffer);
    this._u32 = new Uint32Array(this._f32Buffer);
    
    // Collection of registered layers and operations
    this.ops = {};
    
    // Register built-in operations
    this._registerBuiltinOps();
  }

  // Fast math operations - changed to instance methods
  fastExp(x) {
    let y = 1.0 + x / 256;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    return y;
  }

  fastTanh(x) {
    if (x < -3) return -1;
    if (x > 3) return 1;
    return x * (27 + x * x) / (27 + 9 * x * x);
  }

  fastSigmoid(x) {
    return 1 / (1 + this.fastExp(-x));
  }

  fastInvSqrt(x) {
    const threehalfs = 1.5;
    const x2 = x * 0.5;
    this._f32[0] = x;
    let i = this._u32[0];
    i = 0x5f3759df - (i >> 1);
    this._u32[0] = i;
    let y = this._f32[0];
    y = y * (threehalfs - (x2 * y * y));
    return y;
  }

  _registerBuiltinOps() {
    // Register core tensor operations with loop unrolling
    this.registerOp('add', (a, b, out = null) => {
      const len = a.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = a[i] + b[i];
        out[i+1] = a[i+1] + b[i+1];
        out[i+2] = a[i+2] + b[i+2];
        out[i+3] = a[i+3] + b[i+3];
        out[i+4] = a[i+4] + b[i+4];
        out[i+5] = a[i+5] + b[i+5];
        out[i+6] = a[i+6] + b[i+6];
        out[i+7] = a[i+7] + b[i+7];
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = a[i] + b[i];
      }
      
      return out;
    });

    this.registerOp('sub', (a, b, out = null) => {
      const len = a.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = a[i] - b[i];
        out[i+1] = a[i+1] - b[i+1];
        out[i+2] = a[i+2] - b[i+2];
        out[i+3] = a[i+3] - b[i+3];
        out[i+4] = a[i+4] - b[i+4];
        out[i+5] = a[i+5] - b[i+5];
        out[i+6] = a[i+6] - b[i+6];
        out[i+7] = a[i+7] - b[i+7];
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = a[i] - b[i];
      }
      
      return out;
    });

    this.registerOp('mul', (a, b, out = null) => {
      const len = a.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = a[i] * b[i];
        out[i+1] = a[i+1] * b[i+1];
        out[i+2] = a[i+2] * b[i+2];
        out[i+3] = a[i+3] * b[i+3];
        out[i+4] = a[i+4] * b[i+4];
        out[i+5] = a[i+5] * b[i+5];
        out[i+6] = a[i+6] * b[i+6];
        out[i+7] = a[i+7] * b[i+7];
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = a[i] * b[i];
      }
      
      return out;
    });

    this.registerOp('div', (a, b, out = null) => {
      const len = a.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = a[i] / b[i];
        out[i+1] = a[i+1] / b[i+1];
        out[i+2] = a[i+2] / b[i+2];
        out[i+3] = a[i+3] / b[i+3];
        out[i+4] = a[i+4] / b[i+4];
        out[i+5] = a[i+5] / b[i+5];
        out[i+6] = a[i+6] / b[i+6];
        out[i+7] = a[i+7] / b[i+7];
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = a[i] / b[i];
      }
      
      return out;
    });

    this.registerOp('scalarMul', (a, scalar, out = null) => {
      const len = a.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = a[i] * scalar;
        out[i+1] = a[i+1] * scalar;
        out[i+2] = a[i+2] * scalar;
        out[i+3] = a[i+3] * scalar;
        out[i+4] = a[i+4] * scalar;
        out[i+5] = a[i+5] * scalar;
        out[i+6] = a[i+6] * scalar;
        out[i+7] = a[i+7] * scalar;
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = a[i] * scalar;
      }
      
      return out;
    });

    this.registerOp('scalarAdd', (a, scalar, out = null) => {
      const len = a.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = a[i] + scalar;
        out[i+1] = a[i+1] + scalar;
        out[i+2] = a[i+2] + scalar;
        out[i+3] = a[i+3] + scalar;
        out[i+4] = a[i+4] + scalar;
        out[i+5] = a[i+5] + scalar;
        out[i+6] = a[i+6] + scalar;
        out[i+7] = a[i+7] + scalar;
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = a[i] + scalar;
      }
      
      return out;
    });

    this.registerOp('dot', (a, b) => {
      const len = a.length;
      let sum = 0;
      
      // Already using loop unrolling with 8 elements per iteration
      for (let i = 0; i < len; i += 8) {
        sum += a[i] * b[i] +
               (i+1 < len ? a[i+1] * b[i+1] : 0) +
               (i+2 < len ? a[i+2] * b[i+2] : 0) +
               (i+3 < len ? a[i+3] * b[i+3] : 0) +
               (i+4 < len ? a[i+4] * b[i+4] : 0) +
               (i+5 < len ? a[i+5] * b[i+5] : 0) +
               (i+6 < len ? a[i+6] * b[i+6] : 0) +
               (i+7 < len ? a[i+7] * b[i+7] : 0);
      }
      
      return sum;
    });

    this.registerOp('matmul', (a, b, out = null) => {
      // Assumes a is [m, k] and b is [k, n]
      const m = a.length / a.stride;
      const k = a.stride;
      const n = b.length / k;
      
      out = out || new Float32Array(m * n);
      
      // Loop unrolling not applied here directly as it's more complex
      // Consider tile-based matrix multiplication for better cache performance
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0;
          // Unroll the innermost loop by 4
          let l = 0;
          for (; l <= k - 4; l += 4) {
            sum += a[i * k + l] * b[l * n + j] +
                   a[i * k + l + 1] * b[(l + 1) * n + j] +
                   a[i * k + l + 2] * b[(l + 2) * n + j] +
                   a[i * k + l + 3] * b[(l + 3) * n + j];
          }
          // Handle remaining elements
          for (; l < k; l++) {
            sum += a[i * k + l] * b[l * n + j];
          }
          out[i * n + j] = sum;
        }
      }
      
      return out;
    });

    // Register activation functions
    this.registerOp('relu', (x, out = null) => {
      const len = x.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = Math.max(0, x[i]);
        out[i+1] = Math.max(0, x[i+1]);
        out[i+2] = Math.max(0, x[i+2]);
        out[i+3] = Math.max(0, x[i+3]);
        out[i+4] = Math.max(0, x[i+4]);
        out[i+5] = Math.max(0, x[i+5]);
        out[i+6] = Math.max(0, x[i+6]);
        out[i+7] = Math.max(0, x[i+7]);
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = Math.max(0, x[i]);
      }
      
      return out;
    });

    this.registerOp('tanh', (x, out = null) => {
      const len = x.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = this.fastTanh(x[i]);
        out[i+1] = this.fastTanh(x[i+1]);
        out[i+2] = this.fastTanh(x[i+2]);
        out[i+3] = this.fastTanh(x[i+3]);
        out[i+4] = this.fastTanh(x[i+4]);
        out[i+5] = this.fastTanh(x[i+5]);
        out[i+6] = this.fastTanh(x[i+6]);
        out[i+7] = this.fastTanh(x[i+7]);
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = this.fastTanh(x[i]);
      }
      
      return out;
    });

    this.registerOp('sigmoid', (x, out = null) => {
      const len = x.length;
      out = out || new Float32Array(len);
      
      let i = 0;
      // Process 8 elements per iteration
      for (; i <= len - 8; i += 8) {
        out[i] = this.fastSigmoid(x[i]);
        out[i+1] = this.fastSigmoid(x[i+1]);
        out[i+2] = this.fastSigmoid(x[i+2]);
        out[i+3] = this.fastSigmoid(x[i+3]);
        out[i+4] = this.fastSigmoid(x[i+4]);
        out[i+5] = this.fastSigmoid(x[i+5]);
        out[i+6] = this.fastSigmoid(x[i+6]);
        out[i+7] = this.fastSigmoid(x[i+7]);
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = this.fastSigmoid(x[i]);
      }
      
      return out;
    });

    // Register normalization operations
    this.registerOp('layerNorm', (x, out = null) => {
      const len = x.length;
      out = out || new Float32Array(len);
      
      // Compute mean and variance (already using loop unrolling)
      let sum = 0, sumSq = 0;
      for (let i = 0; i < len; i += 8) {
        const i1 = i, i2 = i+1, i3 = i+2, i4 = i+3, 
              i5 = i+4, i6 = i+5, i7 = i+6, i8 = i+7;
        
        const v1 = i1 < len ? x[i1] : 0;
        const v2 = i2 < len ? x[i2] : 0;
        const v3 = i3 < len ? x[i3] : 0;
        const v4 = i4 < len ? x[i4] : 0;
        const v5 = i5 < len ? x[i5] : 0;
        const v6 = i6 < len ? x[i6] : 0;
        const v7 = i7 < len ? x[i7] : 0;
        const v8 = i8 < len ? x[i8] : 0;
        
        sum += v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
        sumSq += v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7 + v8*v8;
      }
      
      const mean = sum / len;
      const variance = sumSq / len - mean * mean;
      const invStd = this.fastInvSqrt(variance + 1e-5);
      
      // Normalize with loop unrolling
      let i = 0;
      for (; i <= len - 8; i += 8) {
        out[i] = (x[i] - mean) * invStd;
        out[i+1] = (x[i+1] - mean) * invStd;
        out[i+2] = (x[i+2] - mean) * invStd;
        out[i+3] = (x[i+3] - mean) * invStd;
        out[i+4] = (x[i+4] - mean) * invStd;
        out[i+5] = (x[i+5] - mean) * invStd;
        out[i+6] = (x[i+6] - mean) * invStd;
        out[i+7] = (x[i+7] - mean) * invStd;
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] = (x[i] - mean) * invStd;
      }
      
      return out;
    });

    // Register softmax operation
    this.registerOp('softmax', (x, out = null) => {
      const len = x.length;
      out = out || new Float32Array(len);
      
      // Find max for numerical stability
      let max = -Infinity;
      for (let i = 0; i < len; i++) {
        if (x[i] > max) max = x[i];
      }
      
      // Compute exponentials and sum with loop unrolling
      let sum = 0;
      let i = 0;
      for (; i <= len - 4; i += 4) {
        const exp1 = this.fastExp(x[i] - max);
        const exp2 = this.fastExp(x[i+1] - max);
        const exp3 = this.fastExp(x[i+2] - max);
        const exp4 = this.fastExp(x[i+3] - max);
        
        out[i] = exp1;
        out[i+1] = exp2;
        out[i+2] = exp3;
        out[i+3] = exp4;
        
        sum += exp1 + exp2 + exp3 + exp4;
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        const expVal = this.fastExp(x[i] - max);
        out[i] = expVal;
        sum += expVal;
      }
      
      // Normalize with loop unrolling
      const invSum = 1 / sum;
      i = 0;
      for (; i <= len - 8; i += 8) {
        out[i] *= invSum;
        out[i+1] *= invSum;
        out[i+2] *= invSum;
        out[i+3] *= invSum;
        out[i+4] *= invSum;
        out[i+5] *= invSum;
        out[i+6] *= invSum;
        out[i+7] *= invSum;
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        out[i] *= invSum;
      }
      
      return out;
    });

    // Register gradient clipping
    this.registerOp('clipGradients', (grad, maxNorm = 1.0) => {
      let normSquared = 0;
      const len = grad.length;
      
      // Compute norm squared with loop unrolling
      let i = 0;
      for (; i <= len - 8; i += 8) {
        normSquared += grad[i] * grad[i] +
                     grad[i+1] * grad[i+1] +
                     grad[i+2] * grad[i+2] +
                     grad[i+3] * grad[i+3] +
                     grad[i+4] * grad[i+4] +
                     grad[i+5] * grad[i+5] +
                     grad[i+6] * grad[i+6] +
                     grad[i+7] * grad[i+7];
      }
      
      // Handle remaining elements
      for (; i < len; i++) {
        normSquared += grad[i] * grad[i];
      }
      
      if (normSquared > maxNorm * maxNorm) {
        const scale = maxNorm / Math.sqrt(normSquared);
        
        // Apply scaling with loop unrolling
        i = 0;
        for (; i <= len - 8; i += 8) {
          grad[i] *= scale;
          grad[i+1] *= scale;
          grad[i+2] *= scale;
          grad[i+3] *= scale;
          grad[i+4] *= scale;
          grad[i+5] *= scale;
          grad[i+6] *= scale;
          grad[i+7] *= scale;
        }
        
        // Handle remaining elements
        for (; i < len; i++) {
          grad[i] *= scale;
        }
      }
      
      return grad;
    });
  }

  // Register a custom operation
  registerOp(name, fn) {
    this.ops[name] = fn;
  }

  // Create a tensor
  tensor(data, shape = null) {
    if (!(data instanceof Float32Array)) {
      data = new Float32Array(data);
    }
    
    return {
      data,
      shape: shape || [data.length],
      length: data.length,
      get: function(i) { return this.data[i]; },
      set: function(i, v) { this.data[i] = v; }
    };
  }

  // Call an operation
  call(opName, ...args) {
    if (!this.ops[opName]) {
      throw new Error(`Operation ${opName} not found`);
    }
    return this.ops[opName](...args);
  }
}

// Layer base class
class Layer {
  constructor(config = {}) {
    this.config = config;
    this.params = {};
    this.grads = {};
    this.momentum = {};
    this.nn = config.nn || new NanoNN();
    this.initialize();
  }
  
  initialize() {
    // Override in subclasses
  }
  
  forward(input, training = false) {
    // Override in subclasses
    return input;
  }
  
  backward(gradOutput) {
    // Override in subclasses
    return gradOutput;
  }
  
  updateParameters(lr, momentumCoeff = 0.9, gradientThreshold = 0) {
    // Default parameter update with momentum
    for (const key in this.grads) {
      const grad = this.grads[key];
      const param = this.params[key];
      const momentum = this.momentum[key];
      
      for (let i = 0; i < grad.length; i++) {
        if (Math.abs(grad[i]) >= gradientThreshold) {
          momentum[i] = momentumCoeff * momentum[i] + lr * grad[i];
          param[i] -= momentum[i];
        }
        // Reset gradient
        grad[i] = 0;
      }
    }
  }
  
  // Helper to create parameters
  createParam(name, shape, scale = 0.1) {
    const size = Array.isArray(shape) ? shape.reduce((a, b) => a * b, 1) : shape;
    const param = new Float32Array(size);
    
    // Initialize with scaled random values
    for (let i = 0; i < size; i++) {
      param[i] = (Math.random() - 0.5) * scale;
    }
    
    this.params[name] = param;
    this.grads[name] = new Float32Array(size);
    this.momentum[name] = new Float32Array(size);
    
    return param;
  }
}

// Dense/Linear layer implementation
class Dense extends Layer {
  constructor(config) {
    super(config);
  }
  
  initialize() {
    const { inputDim, outputDim, scale = 0.1 } = this.config;
    
    this.createParam('weights', inputDim * outputDim, scale);
    this.createParam('bias', outputDim, scale);
    
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.input = null;
  }
  
  forward(input, training = false) {
    if (training) {
      this.input = new Float32Array(input);
    }
    
    const output = new Float32Array(this.outputDim);
    
    // Perform matrix multiplication: output = input * weights + bias
    for (let i = 0; i < this.outputDim; i++) {
      let sum = this.params.bias[i];
      for (let j = 0; j < this.inputDim; j++) {
        sum += input[j] * this.params.weights[j * this.outputDim + i];
      }
      output[i] = sum;
    }
    
    return output;
  }
  
  backward(gradOutput) {
    const gradInput = new Float32Array(this.inputDim);
    
    // Compute gradients
    for (let i = 0; i < this.outputDim; i++) {
      this.grads.bias[i] += gradOutput[i];
      
      for (let j = 0; j < this.inputDim; j++) {
        this.grads.weights[j * this.outputDim + i] += this.input[j] * gradOutput[i];
        gradInput[j] += this.params.weights[j * this.outputDim + i] * gradOutput[i];
      }
    }
    
    return gradInput;
  }
}

// Layer normalization layer
class LayerNorm extends Layer {
  constructor(config) {
    super(config);
  }
  
  initialize() {
    const { dim } = this.config;
    this.dim = dim;
    this.createParam('gamma', dim, 1.0); // Initially set to ones
    this.createParam('beta', dim, 0.0);  // Initially set to zeros
    
    for (let i = 0; i < dim; i++) {
      this.params.gamma[i] = 1.0;
      this.params.beta[i] = 0.0;
    }
    
    this.input = null;
    this.normalized = null;
    this.mean = 0;
    this.variance = 0;
    this.invStd = 0;
  }
  
  forward(input, training = false) {
    const output = new Float32Array(this.dim);
    
    if (training) {
      this.input = new Float32Array(input);
      this.normalized = new Float32Array(this.dim);
    }
    
    // Compute mean and variance
    let sum = 0, sumSq = 0;
    for (let i = 0; i < this.dim; i++) {
      sum += input[i];
      sumSq += input[i] * input[i];
    }
    
    this.mean = sum / this.dim;
    this.variance = sumSq / this.dim - this.mean * this.mean;
    this.invStd = NanoNN.fastInvSqrt(this.variance + 1e-5);
    
    // Normalize and scale
    for (let i = 0; i < this.dim; i++) {
      const normalized = (input[i] - this.mean) * this.invStd;
      if (training) {
        this.normalized[i] = normalized;
      }
      output[i] = this.params.gamma[i] * normalized + this.params.beta[i];
    }
    
    return output;
  }
  
  backward(gradOutput) {
    const gradInput = new Float32Array(this.dim);
    
    // Compute gradients for gamma and beta
    for (let i = 0; i < this.dim; i++) {
      this.grads.gamma[i] += gradOutput[i] * this.normalized[i];
      this.grads.beta[i] += gradOutput[i];
    }
    
    // Compute gradient with respect to normalized input
    let gradMean = 0;
    let gradVariance = 0;
    
    for (let i = 0; i < this.dim; i++) {
      const gradNormalized = gradOutput[i] * this.params.gamma[i];
      gradInput[i] += gradNormalized * this.invStd;
      gradMean -= gradNormalized * this.invStd;
      gradVariance -= gradNormalized * (this.input[i] - this.mean) * 0.5 * Math.pow(this.invStd, 3);
    }
    
    gradMean /= this.dim;
    gradVariance /= this.dim;
    
    // Add gradients from mean and variance terms
    for (let i = 0; i < this.dim; i++) {
      gradInput[i] += gradMean + 2 * (this.input[i] - this.mean) * gradVariance;
    }
    
    return gradInput;
  }
}

// Embedding layer
class Embedding extends Layer {
  constructor(config) {
    super(config);
  }
  
  initialize() {
    const { vocabSize, embedDim, scale = 0.1 } = this.config;
    
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.createParam('embeddings', vocabSize * embedDim, scale);
    this.tokenId = null;
  }
  
  forward(tokenId, training = false) {
    if (training) {
      this.tokenId = tokenId;
    }
    
    const embedding = new Float32Array(this.embedDim);
    const offset = tokenId * this.embedDim;
    
    for (let i = 0; i < this.embedDim; i++) {
      embedding[i] = this.params.embeddings[offset + i];
    }
    
    return embedding;
  }
  
  backward(gradOutput) {
    const offset = this.tokenId * this.embedDim;
    
    for (let i = 0; i < this.embedDim; i++) {
      this.grads.embeddings[offset + i] += gradOutput[i];
    }
    
    // No gradient to propagate backwards for embeddings
    return null;
  }
}

// Activation layers
class Activation extends Layer {
  constructor(config) {
    super(config);
  }
  
  initialize() {
    this.activationType = this.config.type || 'relu';
    this.input = null;
  }
  
  forward(input, training = false) {
    if (training) {
      this.input = new Float32Array(input);
    }
    
    return this.nn.call(this.activationType, input);
  }
  
  backward(gradOutput) {
    const gradInput = new Float32Array(gradOutput.length);
    
    switch (this.activationType) {
      case 'relu':
        for (let i = 0; i < gradInput.length; i++) {
          gradInput[i] = this.input[i] > 0 ? gradOutput[i] : 0;
        }
        break;
        
      case 'tanh':
        for (let i = 0; i < gradInput.length; i++) {
          const tanh = NanoNN.fastTanh(this.input[i]);
          gradInput[i] = gradOutput[i] * (1 - tanh * tanh);
        }
        break;
        
      case 'sigmoid':
        for (let i = 0; i < gradInput.length; i++) {
          const sigmoid = NanoNN.fastSigmoid(this.input[i]);
          gradInput[i] = gradOutput[i] * sigmoid * (1 - sigmoid);
        }
        break;
        
      default:
        throw new Error(`Unknown activation type: ${this.activationType}`);
    }
    
    return gradInput;
  }
}

// Model class to compose layers
class Model {
  constructor() {
    this.layers = [];
    this.nn = new NanoNN();
    this.compiled = false;
  }
  
  add(layer) {
    if (!layer.nn) {
      layer.nn = this.nn;
    }
    this.layers.push(layer);
    return this;
  }
  
  compile(config = {}) {
    this.learningRate = config.learningRate || 0.01;
    this.momentumCoefficient = config.momentumCoefficient || 0.9;
    this.gradientThreshold = config.gradientThreshold || 0;
    this.compiled = true;
    return this;
  }
  
  forward(input, training = false) {
    let output = input;
    
    for (const layer of this.layers) {
      output = layer.forward(output, training);
    }
    
    return output;
  }
  
  backward(gradOutput) {
    let currentGrad = gradOutput;
    
    for (let i = this.layers.length - 1; i >= 0; i--) {
      currentGrad = this.layers[i].backward(currentGrad);
      if (!currentGrad) break; // Stop if layer doesn't propagate gradients
    }
    
    return currentGrad;
  }
  
  updateParameters() {
    for (const layer of this.layers) {
      layer.updateParameters(
        this.learningRate, 
        this.momentumCoefficient,
        this.gradientThreshold
      );
    }
  }
  
  fit(input, target, config = {}) {
    if (!this.compiled) {
      throw new Error('Model must be compiled before training');
    }
    
    const epochs = config.epochs || 1;
    const batchSize = config.batchSize || 1;
    const verbose = config.verbose !== undefined ? config.verbose : true;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      
      // Basic batch training (simplified)
      const batches = Math.ceil(input.length / batchSize);
      
      for (let batch = 0; batch < batches; batch++) {
        const start = batch * batchSize;
        const end = Math.min(start + batchSize, input.length);
        const batchLoss = this.trainBatch(input.slice(start, end), target.slice(start, end));
        totalLoss += batchLoss;
      }
      
      if (verbose) {
        console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${totalLoss / batches}`);
      }
    }
  }
  
  trainBatch(batchInput, batchTarget) {
    let batchLoss = 0;
    
    for (let i = 0; i < batchInput.length; i++) {
      // Forward pass
      const prediction = this.forward(batchInput[i], true);
      
      // Compute loss and initial gradient (simplified cross-entropy)
      const gradOutput = new Float32Array(prediction.length);
      for (let j = 0; j < prediction.length; j++) {
        if (j === batchTarget[i]) {
          gradOutput[j] = prediction[j] - 1;
          batchLoss -= Math.log(Math.max(prediction[j], 1e-7));
        } else {
          gradOutput[j] = prediction[j];
        }
      }
      
      // Backward pass
      this.backward(gradOutput);
      
      // Update parameters
      this.updateParameters();
    }
    
    return batchLoss;
  }
  
  predict(input) {
    return this.forward(input, false);
  }
  
  // Save the model (simplified)
  save() {
    const modelData = {
      layers: this.layers.map(layer => ({
        type: layer.constructor.name,
        config: layer.config,
        params: {}
      }))
    };
    
    // Save parameters for each layer
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      for (const paramName in layer.params) {
        modelData.layers[i].params[paramName] = Array.from(layer.params[paramName]);
      }
    }
    
    return JSON.stringify(modelData);
  }
  
  // Load the model (simplified)
  static load(jsonString) {
    const modelData = JSON.parse(jsonString);
    const model = new Model();
    
    for (const layerData of modelData.layers) {
      // Instantiate layer based on type
      let layer;
      switch (layerData.type) {
        case 'Dense':
          layer = new Dense(layerData.config);
          break;
        case 'LayerNorm':
          layer = new LayerNorm(layerData.config);
          break;
        case 'Embedding':
          layer = new Embedding(layerData.config);
          break;
        case 'Activation':
          layer = new Activation(layerData.config);
          break;
        // Add other layer types as needed
        default:
          throw new Error(`Unknown layer type: ${layerData.type}`);
      }
      
      // Load parameters
      for (const paramName in layerData.params) {
        layer.params[paramName] = new Float32Array(layerData.params[paramName]);
        layer.grads[paramName] = new Float32Array(layer.params[paramName].length);
        layer.momentum[paramName] = new Float32Array(layer.params[paramName].length);
      }
      
      model.add(layer);
    }
    
    return model;
  }
}

// Benchmark comparing NanoNN vs simpler implementation
function benchmark() {
  console.log("NEURAL NETWORK PERFORMANCE BENCHMARK");
  console.log("====================================");
  
  // Sample data for testing
  const size = 10000;
  const a = new Float32Array(size);
  const b = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    a[i] = Math.random();
    b[i] = Math.random();
  }
  
  // Initialize libraries
  const nano = new NanoNN();
  
  // Benchmark functions
  function benchmarkOperation(name, nanoFn, simpleFn, iterations = 1000) {
    // Warmup
    for (let i = 0; i < 10; i++) {
      nanoFn();
      simpleFn();
    }
    
    // Benchmark NanoNN
    const nanoStart = performance.now();
    for (let i = 0; i < iterations; i++) {
      nanoFn();
    }
    const nanoTime = performance.now() - nanoStart;
    
    // Benchmark simple implementation
    const simpleStart = performance.now();
    for (let i = 0; i < iterations; i++) {
      simpleFn();
    }
    const simpleTime = performance.now() - simpleStart;
    
    // Calculate speedup
    const speedup = simpleTime / nanoTime;
    
    console.log(`${name}:`);
    console.log(`  NanoNN: ${nanoTime.toFixed(2)}ms`);
    console.log(`  Simple: ${simpleTime.toFixed(2)}ms`);
    console.log(`  Speedup: ${speedup.toFixed(2)}x`);
    console.log();
    
    return { nano: nanoTime, simple: simpleTime, speedup };
  }
  
  // Define simple implementations
  const simpleOps = {
    add: (a, b) => {
      const result = new Array(a.length);
      for (let i = 0; i < a.length; i++) {
        result[i] = a[i] + b[i];
      }
      return result;
    },
    
    dot: (a, b) => {
      let sum = 0;
      for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
      }
      return sum;
    },
    
    sigmoid: (x) => {
      const result = new Array(x.length);
      for (let i = 0; i < x.length; i++) {
        result[i] = 1 / (1 + Math.exp(-x[i]));
      }
      return result;
    },
    
    layerNorm: (x) => {
      const result = new Array(x.length);
      
      // Calculate mean
      let sum = 0;
      for (let i = 0; i < x.length; i++) {
        sum += x[i];
      }
      const mean = sum / x.length;
      
      // Calculate variance
      let sumSq = 0;
      for (let i = 0; i < x.length; i++) {
        sumSq += (x[i] - mean) * (x[i] - mean);
      }
      const variance = sumSq / x.length;
      const invStd = 1 / Math.sqrt(variance + 1e-5);
      
      // Normalize
      for (let i = 0; i < x.length; i++) {
        result[i] = (x[i] - mean) * invStd;
      }
      
      return result;
    }
  };
  
  // Run benchmarks
  const results = {
    add: benchmarkOperation(
      "Vector Addition",
      () => nano.call('add', a, b),
      () => simpleOps.add(a, b)
    ),
    
    dot: benchmarkOperation(
      "Dot Product",
      () => nano.call('dot', a, b),
      () => simpleOps.dot(a, b)
    ),
    
    sigmoid: benchmarkOperation(
      "Sigmoid Activation",
      () => nano.call('sigmoid', a),
      () => simpleOps.sigmoid(a)
    ),
    
    layerNorm: benchmarkOperation(
      "Layer Normalization",
      () => nano.call('layerNorm', a),
      () => simpleOps.layerNorm(a)
    )
  };
  
  // Overall summary
  const avgSpeedup = Object.values(results).reduce((sum, r) => sum + r.speedup, 0) / Object.keys(results).length;
  console.log("SUMMARY:");
  console.log(`Average speedup: ${avgSpeedup.toFixed(2)}x`);
  
  return results;
}

// Run the benchmark
benchmark();
