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
        const beg = this._offset;
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
        var buffer_blob = new Blob([subbuffer]);
        var buffer_reader = new FileReader();
        buffer_reader.readAsText(buffer_blob, 'utf-8');
        return buffer_reader.result;
    }
}

tennis.Tensor = class {
    VOID = 0
    constructor(shape, dtype, data) {
        this._shape = []
        this._dtype = this.VOID
        this._data = [] // can be null
        this._value = null  // for js readable value
        this._field = []
    }

    /**
     * @param {[]} fields
     * @return {tennis.Tensor}
     */
    Pack(fields) {
    }
}

tennis.Node = class {
    /**
     * 
     * @param {{}} params 
     */
    constructor(params) {
        for (let param in params) {
            // no use code
            let value = params[param]
        }
        this._params = params
        this._name = this.get("#name", "#name")
        this._op = this.get("#op", "#op")
        this._shape = this.get("#shape")
        this._dtype = this.get("#dtype")
    }

    /**
     * @return {string} name
     */
    name() { return this._name; }

    /**
     * @return {string} op
     */
    op() { return this._op; }

    /**
     * @return {[number]} shape
     */
    shape() { return this._shape; }

    /**
     * @return {number} dtype
     */
    dtype() { return this._dtype; }

    /**
     * 
     * @param {name} param 
     */
    has(param) {
        return param in this._params
    }

    /**
     * 
     * @param {string} param 
     * @param {*} value 
     */
    get(param, value=null) {
        if (param in this._params) {
            return this._params[param]
        }
        return value
    }

    /**
     * @param {[tennis.Node]} nodes 
     * @return {[tennis.Node]}
     */
    inputs(nodes=null) {
        if (nodes) {
            self._inputs = nodes
        } else {
            return self._inputs
        }
    }

    /**
     * 
     * @param {number} i 
     * @return {tennis.Node}
     */
    input(i) {
        return self._inputs[i]
    }
}

tennis.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error running tennis utils.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports = tennis
}
