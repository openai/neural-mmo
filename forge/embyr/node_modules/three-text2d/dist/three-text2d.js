/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};

/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {

/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId])
/******/ 			return installedModules[moduleId].exports;

/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};

/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);

/******/ 		// Flag the module as loaded
/******/ 		module.l = true;

/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}


/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;

/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;

/******/ 	// identity function for calling harmony imports with the correct context
/******/ 	__webpack_require__.i = function(value) { return value; };

/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};

/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};

/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };

/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";

/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 6);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, exports) {

module.exports = THREE;

/***/ }),
/* 1 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var THREE = __webpack_require__(0);
exports.textAlign = {
    center: new THREE.Vector2(0, 0),
    left: new THREE.Vector2(1, 0),
    topLeft: new THREE.Vector2(1, -1),
    topRight: new THREE.Vector2(-1, -1),
    right: new THREE.Vector2(-1, 0),
    bottomLeft: new THREE.Vector2(1, 1),
    bottomRight: new THREE.Vector2(-1, 1),
};
var fontHeightCache = {};
function getFontHeight(fontStyle) {
    var result = fontHeightCache[fontStyle];
    if (!result) {
        var body = document.getElementsByTagName('body')[0];
        var dummy = document.createElement('div');
        var dummyText = document.createTextNode('MÉq');
        dummy.appendChild(dummyText);
        dummy.setAttribute('style', "font:" + fontStyle + ";position:absolute;top:0;left:0");
        body.appendChild(dummy);
        result = dummy.offsetHeight;
        fontHeightCache[fontStyle] = result;
        body.removeChild(dummy);
    }
    return result;
}
exports.getFontHeight = getFontHeight;


/***/ }),
/* 2 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var THREE = __webpack_require__(0);
var utils_1 = __webpack_require__(1);
var CanvasText_1 = __webpack_require__(5);
var Text2D = (function (_super) {
    __extends(Text2D, _super);
    function Text2D(text, options) {
        if (text === void 0) { text = ''; }
        if (options === void 0) { options = {}; }
        var _this = _super.call(this) || this;
        _this._font = options.font || '30px Arial';
        _this._fillStyle = options.fillStyle || '#FFFFFF';
        _this._shadowColor = options.shadowColor || 'rgba(0, 0, 0, 0)';
        _this._shadowBlur = options.shadowBlur || 0;
        _this._shadowOffsetX = options.shadowOffsetX || 0;
        _this._shadowOffsetY = options.shadowOffsetY || 0;
        _this.canvas = new CanvasText_1.CanvasText();
        _this.align = options.align || utils_1.textAlign.center;
        _this.side = options.side || THREE.DoubleSide;
        // this.anchor = Label.fontAlignAnchor[ this._textAlign ]
        _this.antialias = (typeof options.antialias === "undefined") ? true : options.antialias;
        _this.text = text;
        return _this;
    }
    Object.defineProperty(Text2D.prototype, "width", {
        get: function () { return this.canvas.textWidth; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "height", {
        get: function () { return this.canvas.textHeight; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "text", {
        get: function () { return this._text; },
        set: function (value) {
            if (this._text !== value) {
                this._text = value;
                this.updateText();
            }
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "font", {
        get: function () { return this._font; },
        set: function (value) {
            if (this._font !== value) {
                this._font = value;
                this.updateText();
            }
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "fillStyle", {
        get: function () {
            return this._fillStyle;
        },
        set: function (value) {
            if (this._fillStyle !== value) {
                this._fillStyle = value;
                this.updateText();
            }
        },
        enumerable: true,
        configurable: true
    });
    Text2D.prototype.cleanUp = function () {
        if (this.texture) {
            this.texture.dispose();
        }
    };
    Text2D.prototype.applyAntiAlias = function () {
        if (this.antialias === false) {
            this.texture.magFilter = THREE.NearestFilter;
            this.texture.minFilter = THREE.LinearMipMapLinearFilter;
        }
    };
    return Text2D;
}(THREE.Object3D));
exports.Text2D = Text2D;


/***/ }),
/* 3 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var THREE = __webpack_require__(0);
var Text2D_1 = __webpack_require__(2);
var MeshText2D = (function (_super) {
    __extends(MeshText2D, _super);
    function MeshText2D(text, options) {
        if (text === void 0) { text = ''; }
        if (options === void 0) { options = {}; }
        return _super.call(this, text, options) || this;
    }
    MeshText2D.prototype.raycast = function () {
        this.mesh.raycast.apply(this.mesh, arguments);
    };
    MeshText2D.prototype.updateText = function () {
        this.cleanUp(); // cleanup previous texture
        this.canvas.drawText(this._text, {
            font: this._font,
            fillStyle: this._fillStyle,
            shadowBlur: this._shadowBlur,
            shadowColor: this._shadowColor,
            shadowOffsetX: this._shadowOffsetX,
            shadowOffsetY: this._shadowOffsetY,
        });
        this.texture = new THREE.Texture(this.canvas.canvas);
        this.texture.needsUpdate = true;
        this.applyAntiAlias();
        if (!this.material) {
            this.material = new THREE.MeshBasicMaterial({ map: this.texture, side: this.side });
            this.material.transparent = true;
        }
        else {
            this.material.map = this.texture;
        }
        if (!this.mesh) {
            this.geometry = new THREE.PlaneGeometry(this.canvas.width, this.canvas.height);
            this.mesh = new THREE.Mesh(this.geometry, this.material);
            this.add(this.mesh);
        }
        this.mesh.position.x = ((this.canvas.width / 2) - (this.canvas.textWidth / 2)) + ((this.canvas.textWidth / 2) * this.align.x);
        this.mesh.position.y = (-this.canvas.height / 2) + ((this.canvas.textHeight / 2) * this.align.y);
        // manually update geometry vertices
        this.geometry.vertices[0].x = this.geometry.vertices[2].x = -this.canvas.width / 2;
        this.geometry.vertices[1].x = this.geometry.vertices[3].x = this.canvas.width / 2;
        this.geometry.vertices[0].y = this.geometry.vertices[1].y = this.canvas.height / 2;
        this.geometry.vertices[2].y = this.geometry.vertices[3].y = -this.canvas.height / 2;
        this.geometry.verticesNeedUpdate = true;
    };
    return MeshText2D;
}(Text2D_1.Text2D));
exports.MeshText2D = MeshText2D;


/***/ }),
/* 4 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var THREE = __webpack_require__(0);
var Text2D_1 = __webpack_require__(2);
var SpriteText2D = (function (_super) {
    __extends(SpriteText2D, _super);
    function SpriteText2D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    SpriteText2D.prototype.raycast = function () {
        return this.sprite.raycast.apply(this.sprite, arguments);
    };
    SpriteText2D.prototype.updateText = function () {
        this.canvas.drawText(this._text, {
            font: this._font,
            fillStyle: this._fillStyle
        });
        // cleanup previous texture
        this.cleanUp();
        this.texture = new THREE.Texture(this.canvas.canvas);
        this.texture.needsUpdate = true;
        this.applyAntiAlias();
        if (!this.material) {
            this.material = new THREE.SpriteMaterial({ map: this.texture });
        }
        else {
            this.material.map = this.texture;
        }
        if (!this.sprite) {
            this.sprite = new THREE.Sprite(this.material);
            this.geometry = this.sprite.geometry;
            this.add(this.sprite);
        }
        this.sprite.scale.set(this.canvas.width, this.canvas.height, 1);
        this.sprite.position.x = ((this.canvas.width / 2) - (this.canvas.textWidth / 2)) + ((this.canvas.textWidth / 2) * this.align.x);
        this.sprite.position.y = (-this.canvas.height / 2) + ((this.canvas.textHeight / 2) * this.align.y);
    };
    return SpriteText2D;
}(Text2D_1.Text2D));
exports.SpriteText2D = SpriteText2D;


/***/ }),
/* 5 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var THREE = __webpack_require__(0);
var utils_1 = __webpack_require__(1);
var CanvasText = (function () {
    function CanvasText() {
        this.textWidth = null;
        this.textHeight = null;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }
    Object.defineProperty(CanvasText.prototype, "width", {
        get: function () { return this.canvas.width; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(CanvasText.prototype, "height", {
        get: function () { return this.canvas.height; },
        enumerable: true,
        configurable: true
    });
    CanvasText.prototype.drawText = function (text, ctxOptions) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.font = ctxOptions.font;
        this.textWidth = Math.ceil(this.ctx.measureText(text).width);
        this.textHeight = utils_1.getFontHeight(this.ctx.font);
        this.canvas.width = THREE.Math.nextPowerOfTwo(this.textWidth);
        this.canvas.height = THREE.Math.nextPowerOfTwo(this.textHeight);
        this.ctx.font = ctxOptions.font;
        this.ctx.fillStyle = ctxOptions.fillStyle;
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'top';
        this.ctx.shadowColor = ctxOptions.shadowColor;
        this.ctx.shadowBlur = ctxOptions.shadowBlur;
        this.ctx.shadowOffsetX = ctxOptions.shadowOffsetX;
        this.ctx.shadowOffsetY = ctxOptions.shadowOffsetY;
        this.ctx.fillText(text, 0, 0);
        return this.canvas;
    };
    return CanvasText;
}());
exports.CanvasText = CanvasText;


/***/ }),
/* 6 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var SpriteText2D_1 = __webpack_require__(4);
exports.SpriteText2D = SpriteText2D_1.SpriteText2D;
var MeshText2D_1 = __webpack_require__(3);
exports.MeshText2D = MeshText2D_1.MeshText2D;
var utils_1 = __webpack_require__(1);
exports.textAlign = utils_1.textAlign;


/***/ })
/******/ ]);