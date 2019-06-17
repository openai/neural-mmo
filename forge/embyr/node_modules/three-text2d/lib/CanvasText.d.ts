import { TextOptions } from "./Text2D";
export declare class CanvasText {
    textWidth: number;
    textHeight: number;
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    constructor();
    readonly width: number;
    readonly height: number;
    drawText(text: string, ctxOptions: TextOptions): HTMLCanvasElement;
}
