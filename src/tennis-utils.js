/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* TenniS: Tensor based Edge Neural Network Inference System */

var utils = utils || {};
var base = base || require('./base');
var long = long || { Long: require('long') };

utils.Stream = class {
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
            this._dataview = new DataView(data)
            this._offset = offset;
        } else {
            throw utils.Error("Stream param 1 must be DataView or ArrayBuffer")
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
        let high = this.uint32();
        let low = this.uint32();
        return new long.Long(high, low, false).toNumber();
    }

    /**
     * @return {number} 
     */
    uint64() {
        let high = this.uint32();
        let low = this.uint32();
        return new long.Long(high, low, true).toNumber();
    }

    float32() {
        const value = this._dataview.getFloat32(this._offset, true)
        this._offset += 4;
        return value;
    }

    float64() {
        const value = this._dataview.getFloat64(this._offset, true)
        this._offset += 8;
        return value;
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
    int16() {
        const value = this._dataview.getInt16(this._offset, true)
        this._offset += 2;
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
     * @return {number}
     */
    uint8() {
        const value = this._dataview.getUint8(this._offset, true)
        this._offset += 1;
        return value;
    }

    /**
     * @return {number}
     */
    uint16() {
        const value = this._dataview.getUint16(this._offset, true)
        this._offset += 2;
        return value;
    }

    /**
     * @return {number}
     */
    uint32() {
        const value = this._dataview.getUint32(this._offset, true)
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

utils.dtype = class {
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
        throw utils.Error("Not support dtype = " + dtype)
    }

    /**
     * 
     * @param {number} dtype
     * @return {string} 
     */
    static type_str(dtype) {
        switch (dtype) {
            case this.VOID: return "void";
            case this.INT8: return "int8";
            case this.UINT8: return "uint8";
            case this.INT16: return "int16";
            case this.UINT16: return "uint16";
            case this.INT32: return "int32";
            case this.UINT32: return "uint32";
            case this.INT64: return "int64";
            case this.UINT64: return "uint64";
            case this.FLOAT16: return "float16";
            case this.FLOAT32: return "float32";
            case this.FLOAT64: return "float64";
            case this.CHAR8: return "char8";
            case this.CHAR16: return "char16";
            case this.CHAR32: return "char32";
            case this.UNKNOWN8: return "unknown8";
            case this.UNKNOWN16: return "unknown16";
            case this.UNKNOWN32: return "unknown32";
            case this.UNKNOWN64: return "unknown64";
            case this.UNKNOWN128: return "unknown128";
            case this.BOOLEAN: return "bool";
            case this.COMPLEX32: return "complex32";
            case this.COMPLEX64: return "complex64";
            case this.COMPLEX128: return "complex128";
            default: return "unkown";
        }
    }
}

utils.Tensor = class {
    /**
     * 
     * @param {[number]} shape 
     * @param {number} dtype 
     * @param {ArrayBuffer} data 
     * @param {object} value 
     */
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

    get shape() {
        return this._shape;
    }

    get dtype() {
        return this._dtype;
    }

    get count() {
        let prod = 1;
        for (const i of this._shape) {
            prod *= i;
        }
        return prod;
    }

    get proto() {
        return new utils.Prototype(this._dtype, this._shape);
    }
    /**
     * 
     * @param {Function} decoder 
     */
    _decode_core(decoder) {
        if (this._shape.length == 0) {
            return decoder();
        }
        const count = this.count;
        let list = [];
        for (let i = 0; i < count; ++i) {
            list.push(decoder())
        }
        return list;
    }

    _decode() {
        let stream = new utils.Stream(this._data);
        switch (this._dtype) {
        case utils.dtype.INT8:
            return this._decode_core(function() {return stream.int8(); });
        case utils.dtype.INT16:
            return this._decode_core(function() {return stream.int16(); });
        case utils.dtype.INT32:
            return this._decode_core(function() {return stream.int32(); });
        case utils.dtype.INT64:
            return this._decode_core(function() {return stream.int64(); });
        case utils.dtype.UINT8:
            return this._decode_core(function() {return stream.uint8(); });
        case utils.dtype.UINT16:
            return this._decode_core(function() {return stream.uint16(); });
        case utils.dtype.UINT32:
            return this._decode_core(function() {return stream.uint32(); });
        case utils.dtype.UINT64:
            return this._decode_core(function() {return stream.uinst64(); });
        case utils.dtype.FLOAT32:
            return this._decode_core(function() {return stream.float32(); });
        case utils.dtype.FLOAT64:
            return this._decode_core(function() {return stream.float64(); });
        case utils.dtype.BOOLEAN:
            return this._decode_core(function() {return stream.int8() ? true : false; });
        }
        return null;
    }

    get value() {
        if (!(this._value === null)) {
            return this._value;
        }
        switch (this._dtype) {
        case utils.dtype.CHAR8:
            {
                this._value = String.fromCharCode.apply(null, new Int8Array(this._data));
                break;
            }
        default:
            {
                this._value = this._decode();
            }
        }
        return this._value;
    }

    _context() {
        let context = {};
        context.stream = new utils.Stream(this._data);
        let decoder = null;
        switch (this._dtype) {
            case utils.dtype.INT8:
                decoder = function(stream) {return stream.int8(); }; break;
            case utils.dtype.INT16:
                decoder = function(stream) {return stream.int16(); }; break;
            case utils.dtype.INT32:
                decoder = function(stream) {return stream.int32(); }; break;
            case utils.dtype.INT64:
                decoder = function(stream) {return stream.int64(); }; break;
            case utils.dtype.UINT8:
                decoder = function(stream) {return stream.uint8(); }; break;
            case utils.dtype.UINT16:
                decoder = function(stream) {return stream.uint16(); }; break;
            case utils.dtype.UINT32:
                decoder = function(stream) {return stream.uint32(); }; break;
            case utils.dtype.UINT64:
                decoder = function(stream) {return stream.uint64(); }; break;
            case utils.dtype.FLOAT32:
                decoder = function(stream) {return stream.float32(); }; break;
            case utils.dtype.FLOAT64:
                decoder = function(stream) {return stream.float64(); }; break;
            case utils.dtype.BOOLEAN:
                decoder = function(stream) {return stream.uint8(); }; break;
            }
        if (decoder === null) {
            return null;
        }
        context.shape = this._shape;
        context.next = decoder;
        context.count = 0;
        context.limit = Number.MAX_SAFE_INTEGER;

        return context;
    }

    get viewable() {
        return (false ||
            this._dtype == utils.dtype.INT8 ||
            this._dtype == utils.dtype.INT16 ||
            this._dtype == utils.dtype.INT32 ||
            this._dtype == utils.dtype.INT64 ||
            this._dtype == utils.dtype.UINT8 ||
            this._dtype == utils.dtype.UINT16 ||
            this._dtype == utils.dtype.UINT32 ||
            this._dtype == utils.dtype.UINT64 ||
            this._dtype == utils.dtype.FLOAT32 ||
            this._dtype == utils.dtype.FLOAT64 ||
            this._dtype == utils.dtype.BOOLEAN);
    }

    _view(context, dim=0) {
        if (dim >= context.shape.length) {
            return context.next(context.stream);
        }
        let array = [];
        const size = context.shape[dim];
        if (dim == context.shape.length - 1) {
            for (let i = 0; i < size; ++i) {
                if (context.count >= context.limit) {
                    array.push("...");
                    return array;
                }
                array.push(context.next(context.stream));
                context.count++;
            }
        } else {
            for (let i = 0; i < size; ++i) {
                if (context.count >= context.limit) {
                    array.push("...");
                    return array;
                }
                array.push(this._view(context, dim + 1))
            }
        }
        return array;
    }

    view(limit=null) {
        let context = this._context();
        if (context === null) return null;
        if (!(limit === null)) {
            context.limit = limit;
        }
        return this._view(context);
    }

    /**
     * @param {[utils.Tensor]} fields
     * @return {utils.Tensor}
     */
    static Pack(fields) {
        if (fields.length == 0) {
            new utils.Tensor([], utils.dtype.VOID, null);
        } else if (fields.length == 1) {
            return fields[0];
        }
        let extra = [];
        for (let i = 1; i < fields.length; ++i) {
            extra.push(fields[i]);
        }
        let packed = new utils.Tensor(this._shape, this._dtype, this._data, this._value);
        packed._field = extra;
        return packed;
    }
}

utils.Node = class {
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
        this._outputs = [this]

        if (!(this._dtype === null)) {
            this._dtype = this._dtype.value;
        }

        if (!(this._shape === null)) {
            this._shape = this._shape.value;
        }
    }

    get proto() {
        if (this._dtype === null && this._shape === null) {
            return null;
        }
        return new utils.Prototype(this._dtype, this._shape);
    }

    /**
     * @return {string} arg_id
     */
    get arg_id() {
        return this._id.toString() + ": " + this._name;
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
     * @return {{}} params
     */
    get params() { return this._params; }

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
     * @return {[utils.Node]}
     */
    get inputs() {
        return this._inputs;
    }

    /**
     * @param {[utils.Node]} nodes 
     */
    set inputs(v) {
        this._inputs = v;
    }

    /**
     * @return {[utils.Node]}
     */
    get outputs() {
        return this._outputs;
    }

    /**
     * 
     * @param {number} i 
     * @return {utils.Node}
     */
    input(i) {
        return this._inputs[i];
    }
    /**
     * 
     * @param {number} i 
     * @return {utils.Node}
     */
    output(i) {
        return this._outputs[i];
    }
}

utils.Prototype =  class {
    /**
     * 
     * @param {number} dtype 
     * @param {[number]} shape 
     */
    constructor(dtype, shape) {
        this._dtype = dtype;
        this._shape = shape;
    }

    toString() {
        let part1 = this._dtype === null ? "" : utils.dtype.type_str(this._dtype);
        let part2 = "";
        if (this._shape === null) {
            part2 = part1.length > 0 ? "[...]" : "tensor";
        } else if (this._shape.length > 0) {
            part2 += "[";
            for (let i = 0; i < this._shape.length; ++i) {
                if (i > 0) part2 += ", ";
                let dim = this._shape[i];
                if (dim < 0) {
                    part2 += "?";
                } else {
                    part2 += dim;
                }
            }
            part2 += "]";
        }
        return part1 + part2;
    }
}

utils.Module = class {
    /**
     * 
     * @param {utils.Stream} stream 
     */
    constructor(stream) {
        stream.skip(4);
        let mask = stream.int32();
        if (mask != 0x19910929) {
            throw utils.Error("TenniS Module not valid.");
        }
        this._mask = mask;
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
     * @return {number}
     */
    get mask() { return this._mask; }

    /**
     * @return {[utils.Node]} inputs
     */
    get inputs() { return this._inputs; }

    /**
     * @return {[utils.Node]} outputs
     */
    get outputs() { return this._outputs; }

    /**
     * @return {[utils.Node]} nodes
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
     * @param {utils.Stream} stream
     * @return {utils.Tensor} 
     */
    _read_tensor(stream) {
        let size = stream.int32();
        let fileds = [];
        for (let i = 0; i < size; ++i) {
            let dtype = stream.int8();
            let shape = stream.int32_array();
            let data_size = utils.dtype.type_bytes(dtype) * this._prod(shape);
            let data = stream.buffer(data_size);
            let tensor = new utils.Tensor(shape, dtype, data);
            fileds.push(tensor);
        }
        return utils.Tensor.Pack(fileds);
    }

    /**
     * 
     * @param {utils.Stream} stream
     * @param {number} id
     * @return {utils.Node} 
     */
    _read_bubble(stream, id=null) {
        let size = stream.int32();
        let params = {};
        for (let i = 0; i < size; ++i) {
            let name = stream.string();
            let value = this._read_tensor(stream);
            params[name] = value;
        }
        return new utils.Node(params, id);
    }

    /**
     * 
     * @param {utils.Stream} stream 
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

utils.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error running tennis utils.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports = utils;
}
