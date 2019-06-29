import THREE = require("three");
import { Text2D } from "./Text2D";
export declare class MeshText2D extends Text2D {
    mesh: THREE.Mesh;
    protected geometry: THREE.PlaneGeometry;
    constructor(text?: string, options?: {});
    raycast(): void;
    updateText(): void;
}
