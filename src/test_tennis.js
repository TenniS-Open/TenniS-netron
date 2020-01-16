/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* TenniS: Tensor based Edge Neural Network Inference System */

// var tennis = tennis || require("./tennis")
var ts = ts || require("./tennis-utils")
var fs = fs || require("fs")


const filepath = "/Users/seetadev/Documents/SDK/CLion/TensorStack/python/test/0.add.tsm"
var data = fs.readFileSync(filepath)

let stream = new ts.Stream(data)
stream.skip(4)
let mask = stream.int32()
console.log(mask.toString(16))

node = new ts.Node({"a":1})
console.log(node.has("a"))

console.log("+-----------+")
console.log("| All done. |")
console.log("+-----------+")
