export function toRgba(col: string, alpha: number) {
    if (col.startsWith("#")) {
        const [r, g, b] = col.match(/\w\w/g).map((x) => parseInt(x, 16));
        return `rgba(${r},${g},${b},${alpha})`;
    }
    else if (col.startsWith("rgb("))
        return col.replace(")", `, ${alpha})`).replace("rgb", "rgba");
}

export function generateRandomColor() {
    const letters = "0123456789ABCDEF";
    let color = "#";
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}