import THREE = require("three");
import { CanvasText } from "./CanvasText";
export interface TextOptions {
    font?: string;
    fillStyle?: string;
    align?: THREE.Vector2;
    side?: number;
    antialias?: boolean;
    shadowColor?: string;
    shadowBlur?: number;
    shadowOffsetX?: number;
    shadowOffsetY?: number;
}
export declare abstract class Text2D extends THREE.Object3D {
    align: THREE.Vector2;
    side: number;
    antialias: boolean;
    texture: THREE.Texture;
    material: THREE.MeshBasicMaterial | THREE.SpriteMaterial;
    protected _font: string;
    protected _fillStyle: string;
    protected _text: string;
    protected _shadowColor: string;
    protected _shadowBlur: number;
    protected _shadowOffsetX: number;
    protected _shadowOffsetY: number;
    protected canvas: CanvasText;
    protected geometry: THREE.Geometry | THREE.BufferGeometry;
    constructor(text?: string, options?: TextOptions);
    abstract raycast(): void;
    abstract updateText(): void;
    readonly width: number;
    readonly height: number;
    text: string;
    font: string;
    fillStyle: string;
    cleanUp(): void;
    applyAntiAlias(): void;
}
