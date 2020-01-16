/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* TenniS: Tensor based Edge Neural Network Inference System */

var tennis = tennis || {};
var base = base || require('./base');
var long = long || { Long: require('long') };

tennis.Stream = class {
    /**
     * 
     * @param {DataView|ArrayBuffer} data 
     * @param {number} offset
     */
    constructor(data, offset=0) {
        if (data instanceof DataView) {
            this._dataview = data;
            this._offset = offset;
        } else if (data instanceof Buffer) {
            this._dataview = new DataView(data.buffer, data.byteOffset, data.length)
            this._offset = offset;
        } else if (data instanceof ArrayBuffer ||
                data instanceof SharedArrayBuffer) {
            this._dataview = new DataView(data, data.byteLength, data.length)
            this._offset = offset;
        } else {
            throw tennis.Error("Stream param 1 must be DataView or ArrayBuffer")
        }
    }

    /**
     * 
     */
    rewind() {
        this._offset = 0;
    }

    /**
     * @return {number} 
     */
    int64() {
        let high = this.int32();
        let low = this.int32();
        return new long.Long(high, low, true).toNumber();
    }

    /**
     * @return {number}
     */
    offset() {
        return this._offset;
    }

    /**
     * @return {number}
     */
    int8() {
        const value = this._dataview.getInt8(this._offset, true)
        this._offset += 1;
        return value;
    }

    /**
     * @return {number}
     */
    int32() {
        const value = this._dataview.getInt32(this._offset, true)
        this._offset += 4;
        return value;
    }

    /**
     * 
     * @param {number} size 
     * @return {ArrayBuffer}
     */
    buffer(size) {
        let data = this._dataview;
        const beg = data.byteOffset + this._offset;
        const end = beg + size;
        this._offset += size;
        return this._dataview.buffer.slice(beg, end);
    }

    /**
     * @param {number} size
     */
    skip(size) {
        this._offset += size;
    }

    /**
     * @return {string}
     */
    string() {
        const length = this.int32();
        const subbuffer = this.buffer(length);
        let name = String.fromCharCode.apply(null, new Int8Array(subbuffer));
        return name;
        // var buffer_blob = new Blob([subbuffer]);
        // var buffer_reader = new FileReader();
        // buffer_reader.readAsText(buffer_blob, 'utf-8');
        // return buffer_reader.result;
    }

    /**
     * @return {[number]}
     */
    int32_array() {
        const size = this.int32();
        let array = [];
        for (let i = 0; i < size; ++i) {
            array.push(this.int32());
        }
        return array;
    }
}

tennis.dtype = class {
    static VOID = 0;
    static INT8 = 1;
    static UINT8 = 2;
    static INT16 = 3;
    static UINT16 = 4;
    static INT32 = 5;
    static UINT32 = 6;
    static INT64 = 7;
    static UINT64 = 8;
    static FLOAT16 = 9;
    static FLOAT32 = 10;
    static FLOAT64 = 11;
    static PTR = 12;
    static CHAR8 = 13;
    static CHAR16 = 14;
    static CHAR32 = 15;
    static UNKNOWN8 = 16;
    static UNKNOWN16 = 17;
    static UNKNOWN32 = 18;
    static UNKNOWN64 = 19;
    static UNKNOWN128 = 20;
    static BOOLEAN = 21;
    static COMPLEX32 = 22;
    static COMPLEX64 = 23;
    static COMPLEX128 = 24;

    /**
     * 
     * @param {number} dtype
     * @return {number} 
     */
    static type_bytes(dtype) {
        switch (dtype) {
            case this.VOID: return 0;
            case this.INT8: return 1;
            case this.UINT8: return 1;
            case this.INT16: return 2;
            case this.UINT16: return 2;
            case this.INT32: return 4;
            case this.UINT32: return 4;
            case this.INT64: return 8;
            case this.UINT64: return 8;
            case this.FLOAT16: return 2;
            case this.FLOAT32: return 4;
            case this.FLOAT64: return 8;
            case this.CHAR8: return 1;
            case this.CHAR16: return 2;
            case this.CHAR32: return 4;
            case this.UNKNOWN8: return 1;
            case this.UNKNOWN16: return 2;
            case this.UNKNOWN32: return 4;
            case this.UNKNOWN64: return 8;
            case this.UNKNOWN128: return 16;
            case this.BOOLEAN: return 1;
            case this.COMPLEX32: return 4;
            case this.COMPLEX64: return 8;
            case this.COMPLEX128: return 16;
            default:
                break;
        }
        throw tennis.Error("Not support dtype = " + dtype)
    }
}

tennis.Tensor = class {
    constructor(shape, dtype, data, value=null) {
        this._shape = shape
        this._dtype = dtype;
        this._data = data // can be null
        this._field = []
        
        if (value) {
            this._value = value;
        } else {
            this._value = null  // for js readable value
        }
    }

    get count() {
        let prod = 1;
        for (const i of this._shape) {
            prod *= i;
        }
        return prod;
    }

    get value() {
        if (!(this._value === null)) {
            return this._value;
        }
        switch (this._dtype) {
        case tennis.dtype.CHAR8:
            {
                this._value = String.fromCharCode.apply(null, new Int8Array(this._data));
                break;
            }
        }
        return this._value;
    }

    /**
     * @param {[tennis.Tensor]} fields
     * @return {tennis.Tensor}
     */
    static Pack(fields) {
        if (fields.length == 0) {
            new tennis.Tensor([], tennis.dtype.VOID, null);
        } else if (fields.length == 1) {
            return fields[0];
        }
        let extra = [];
        for (let i = 1; i < fields.length; ++i) {
            extra.push(fields[i]);
        }
        let packed = new tennis.Tensor(this._shape, this._dtype, this._data, this._value);
        packed._field = extra;
        return packed;
    }
}

tennis.Node = class {
    /**
     * 
     * @param {{}} params 
     */
    constructor(params, id=null) {
        for (let param in params) {
            // no use code
            let value = params[param];
        }
        this._id = id;
        this._params = params;
        this._name = this.get("#name").value;
        this._op = this.get("#op").value;
        this._shape = this.get("#shape");
        this._dtype = this.get("#dtype");
        this._inputs = []
    }

    /**
     * @return {string} arg_id
     */
    get arg_id() {
        return this._id.toString() + ":" + this._name;
    }

    /**
     * @return {number} id
     */
    get id() {
        return this._id;
    }

    /**
     * @param {number} v
     */
    set id(v) {
        this._id = v;
    }

    /**
     * @return {string} name
     */
    get name() { return this._name; }

    /**
     * @return {string} op
     */
    get op() { return this._op; }

    /**
     * @return {[number]} shape
     */
    get shape() { return this._shape; }

    /**
     * @return {number} dtype
     */
    get dtype() { return this._dtype; }

    /**
     * 
     * @param {name} param 
     */
    has(param) {
        return param in this._params;
    }

    /**
     * 
     * @param {string} param 
     * @param {*} value 
     */
    get(param, value=null) {
        if (param in this._params) {
            return this._params[param];
        }
        return value;
    }

    /**
     * @return {[tennis.Node]}
     */
    get inputs() {
        return this._inputs;
    }

    /**
     * @param {[tennis.Node]} nodes 
     */
    set inputs(v) {
        this._inputs = v;
    }

    /**
     * 
     * @param {number} i 
     * @return {tennis.Node}
     */
    input(i) {
        return this._inputs[i];
    }
}

tennis.Module = class {
    /**
     * 
     * @param {tennis.Stream} stream 
     */
    constructor(stream) {
        stream.skip(4);
        let mask = stream.int32();
        if (mask != 0x19910929) {
            throw tennis.Error("TenniS Module not valid.");
        }
        stream.skip(120);
        let inputs = stream.int32_array();
        let outputs = stream.int32_array();
        this._nodes = [];
        this._read_graph(stream);
        this._inputs = []
        this._outputs = []
        for (const i of inputs) {
            this._inputs.push(this._nodes[i]);
        }
        for (const i of outputs) {
            this._outputs.push(this._nodes[i]);
        }
    }

    /**
     * @return {[tennis.Node]} inputs
     */
    get inputs() { return this._inputs; }

    /**
     * @return {[tennis.Node]} outputs
     */
    get outputs() { return this._outputs; }

    /**
     * @return {[tennis.Node]} nodes
     */
    get nodes() { return this._nodes; }

    /**
     * @param {[number]} shape
     * @return {number}
     */
    _prod(shape) {
        let prod = 1;
        for (const i of shape) {
            prod *= i;
        }
        return prod;
    }

    /**
     * 
     * @param {tennis.Stream} stream
     * @return {tennis.Tensor} 
     */
    _read_tensor(stream) {
        let size = stream.int32();
        let fileds = [];
        for (let i = 0; i < size; ++i) {
            let dtype = stream.int8();
            let shape = stream.int32_array();
            let data_size = tennis.dtype.type_bytes(dtype) * this._prod(shape);
            let data = stream.buffer(data_size);
            let tensor = new tennis.Tensor(shape, dtype, data);
            fileds.push(tensor);
        }
        return tennis.Tensor.Pack(fileds);
    }

    /**
     * 
     * @param {tennis.Stream} stream
     * @param {number} id
     * @return {tennis.Node} 
     */
    _read_bubble(stream, id=null) {
        let size = stream.int32();
        let params = {};
        for (let i = 0; i < size; ++i) {
            let name = stream.string();
            let value = this._read_tensor(stream);
            params[name] = value;
        }
        return new tennis.Node(params, id);
    }

    /**
     * 
     * @param {tennis.Stream} stream 
     */
    _read_graph(stream) {
        let size = stream.int32();
        let nodes = [];
        let node_inputs = [];
        for (let i = 0; i < size; ++i) {
            nodes.push(this._read_bubble(stream, i));
            node_inputs.push(stream.int32_array());
        }
        for (let i = 0; i < size; ++i) {
            let inputs = [];
            for (const j of node_inputs[i]) {
                inputs.push(nodes[j])
            }
            nodes[i].inputs = inputs;
        }
        this._nodes = nodes;
    }
};

tennis.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error running tennis utils.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports = tennis
}
