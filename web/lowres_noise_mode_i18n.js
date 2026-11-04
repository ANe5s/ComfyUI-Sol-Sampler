import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "SolAttnProgressiveSampler";
const INPUT_NAME = "lowres_noise_mode";

let translations = {};

api.getCustomNodesI18n().then((data) => {
    translations = data ?? {};
    for (const node of app.graph?._nodes ?? []) {
        if ((node.comfyClass ?? node.constructor?.comfyClass) === NODE_TYPE) {
            node.setDirtyCanvas?.(true, true);
        }
    }
}).catch(() => {});

function optionLabels() {
    const locale = String(app.ui.settings.getSettingValue("Comfy.Locale") || "en").toLowerCase();
    const localeKey = locale.startsWith("zh") ? "zh" : locale;
    return translations[localeKey]?.nodeDefs?.[NODE_TYPE]?.inputs?.[INPUT_NAME]?.options ?? {};
}

app.registerExtension({
    name: "ANe5s.SolSampler.NoiseModeI18n",
    nodeCreated(node) {
        if ((node.comfyClass ?? node.constructor?.comfyClass) !== NODE_TYPE) return;

        const widget = node.widgets?.find((item) => item.name === INPUT_NAME);
        if (!widget) return;

        const originalGetOptionLabel = widget.options?.getOptionLabel;
        widget.options ??= {};
        widget.options.getOptionLabel = (value) => {
            const key = String(value ?? "");
            return optionLabels()[key] ?? originalGetOptionLabel?.(value) ?? key;
        };
    },
});
