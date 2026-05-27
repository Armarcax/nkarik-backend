// Shared Data for Nkarik AI

const UI_STRINGS = {
            hy: {
                pencil: "Մատիտ", brush: "Վրձին", pen: "Գրիչ", eraser: "Ռետին", clear: "Ջնջել", processing: "Մշակվում է...",
                alertFail: "Հարցումը ձախողվեց: Խնդրում ենք փորձել կրկին:", voicePlaceholder: "Գրիր այստեղ կամ խոսիր միկրոֆոնով..."
            },
            en: {
                pencil: "Pencil", brush: "Brush", pen: "Pen", eraser: "Eraser", clear: "Clear", processing: "Processing...",
                alertFail: "Request failed. Please try again!", voicePlaceholder: "Type here or speak to mic..."
            },
            ru: {
                pencil: "Карандаш", brush: "Кисть", pen: "Ручка", eraser: "Ластик", clear: "Стереть", processing: "Магия...",
                alertFail: "Ошибка соединения. Попробуйте еще раз!", voicePlaceholder: "Пишите здесь или говорите в микрофон..."
            }
        };

const DICTIONARY_MAP = {
            "խնձոր": "apple", "ձուկ": "fish", "ձկնիկ": "cute fish", "կատու": "cat", "շուն": "dog", "շունիկ": "puppy dog",
            "մեքենա": "car", "ավտո": "car", "տուն": "house", "տնակ": "cozy little house", "ծառ": "tree", "արև": "sun",
            "ամպ": "cloud", "ամպիկ": "fluffy cloud", "թիթեռ": "butterfly", "ծաղիկ": "flower", "նապաստակ": "rabbit",
            "արջուկ": "teddy bear", "գնդակ": "ball", "նավակ": "boat", "աստղ": "star", "լուսին": "moon",
            "հեծանիվ": "bicycle", "աղվես": "fox", "առյուծ": "lion", "փիղ": "elephant", "ձի": "horse",
            "яблоко": "apple", "рыба": "fish", "рыбка": "cute fish", "кошка": "cat", "кот": "cat", "собака": "dog",
            "собачка": "puppy dog", "машина": "car", "дом": "house", "домик": "cozy little house", "дерево": "tree",
            "солнышко": "sun", "солнце": "sun", "облако": "cloud", "облачко": "fluffy cloud", "бабочка": "butterfly",
            "цветок": "flower", "зайчик": "rabbit", "заяц": "rabbit", "мишка": "teddy bear", "медведь": "bear",
            "мяч": "ball", "кораблик": "boat", "лодка": "boat", "звезда": "star", "луна": "moon", "лиса": "fox"
        };

const ART_STYLES = [
            // --- SIMPLE CARTOON / KIDS STYLES ---
            { id: 'cartoon', label: 'Cartoon 🧸', prompt: 'cute cheerful cartoon style simple lines' },
            { id: 'crayon', label: 'Crayon 🖍️', prompt: 'bright child crayon drawing whimsical scribbles' },
            { id: 'sticker', label: 'Sticker ✨', prompt: 'cute die-cut sticker graphic white border vector format' },
            { id: 'claymation', label: 'Claymation 🎭', prompt: 'cute plasticine clay model animation character design' },
            { id: 'pixel', label: 'Pixel 👾', prompt: 'retro 8bit clean cute pixel art flat icon asset' },
            { id: 'anime', label: 'Anime ⚡', prompt: 'studio ghibli style anime cheerful fantasy artwork' },
            { id: 'coloring_book', label: 'Coloring 📖', prompt: 'bold thick black outline coloring book page for toddlers' },
            { id: 'flat_vector', label: 'Flat Vector 🦄', prompt: 'simple minimalist flat vector art bright solid shapes' },
            { id: 'felt_craft', label: 'Felt Craft 🧵', prompt: 'stitched colorful felt fabric cutout craft layout' },
            { id: 'chalk', label: 'Chalk Art ⬜', prompt: 'blackboard school chalk colorful basic drawings sketch' },
            { id: 'glow_dark', label: 'Glow Dark 🌙', prompt: 'bioluminescent neon glowing magical fantasy graphics for children' },
            { id: 'low_poly', label: 'Low Poly 💎', prompt: 'vivid modern simple low-poly 3d vector graphic asset' },
            { id: 'kawaii', label: 'Kawaii 🌸', prompt: 'ultra cute Japanese kawaii style illustration soft pastel tones' },
            { id: 'doodle', label: 'Doodle ✏️', prompt: 'playful quirky hand-drawn line doodle concept happy vibe' },
            { id: 'paper_cutout', label: 'Papercraft ✂️', prompt: '3D layered multi-colored construction paper cut craft layout' },

            // --- DETAILED / ADULT / REALISTIC STYLES ---
            { id: 'realistic', label: 'Realistic 📸', prompt: 'clean vivid professional photography commercial lighting portrait' },
            { id: 'watercolor', label: 'Watercolor 🎨', prompt: 'soft fine art detailed pastel watercolor painting illustration' },
            { id: 'oil_paint', label: 'Oil Painting 🖼️', prompt: 'vibrant texturized professional impasto oil painting masterpiece artwork' },
            { id: 'charcoal', label: 'Charcoal 🖤', prompt: 'soft fine architectural charcoal blended artistic dramatic sketch' },
            { id: 'pencil_sketch', label: 'Sketch 📝', prompt: 'highly detailed fine pencil artistic precise graphite drawing concept' },
            { id: 'cyberpunk', label: 'Cyberpunk 🌆', prompt: 'intricate cinematic hyper-detailed futuristic sci-fi neon artwork' },
            { id: 'popart', label: 'Pop Art 🍿', prompt: 'bold professional pop art comic style lines halftone textures' },
            { id: 'mosaic', label: 'Mosaic 🧩', prompt: 'ancient complex colorful decorative ceramic tile mosaic layout pattern' },
            { id: 'origami', label: 'Orig Origami 📐', prompt: 'geometric complex folded Japanese origami paper structural craft design' },
            { id: 'digital_airbrush', label: 'Airbrush 💨', prompt: 'smooth hyper-vivid glossy industrial airbrush fine illustration' },
            { id: 'psychedelic', label: 'Vivid Retro 🍭', prompt: 'complex whimsical 70s colorful pop design surreal format' },
            { id: 'stained_glass', label: 'Glass Art ⛪', prompt: 'vibrant custom stained glass window graphic segments gothic luxury' },
            { id: 'abstract', label: 'Abstract 🌀', prompt: 'vivid dynamic non-representational creative museum modern expressionism' },
            { id: 'van_gogh', label: 'Van Gogh 🌻', prompt: 'starry night dynamic sweeping heavy brushstrokes impressionism theme' },
            { id: 'isometric', label: '3D Block 🧊', prompt: 'detailed dynamic architectural isometric 3d voxel block asset' },
            { id: 'u_e', label: 'Ukiyo-e 🌊', prompt: 'classic Japanese traditional woodblock professional art print style' },
            { id: 'futuristic', label: 'Futuristic 🚀', prompt: 'clean high-tech space concept illustration digital design layout' },
            { id: 'vintage_comic', label: 'Retro Comic 📚', prompt: 'vintage golden-age comic book print ink dots structural texture' },
            { id: 'gothic_oil', label: 'Gothic Dark 🏰', prompt: 'moody dramatic atmospheric classical oil painting dark academic style' },
            { id: 'art_deco', label: 'Art Deco ⚜️', prompt: '1920s luxurious geometric ornament gold and sleek lines illustration' },
            { id: 'steampunk', label: 'Steampunk ⚙️', prompt: 'industrial brass gears intricate machinery Victorian engine concept sketch' }
        ];

const GUIDED_TEMPLATES = [
        // --- OBJECTS ---
        { id: "apple", cat: "objects", label: { hy: "Խնձոր", en: "Apple", ru: "Яблоко" }, svg: `<svg viewBox="0 0 100 100"><path d="M50,25 C40,25 30,30 30,45 C30,65 45,85 50,85 C55,85 70,65 70,45 C70,30 60,25 50,25 Z" /><path d="M50,25 Q55,15 60,10" /></svg>` },
        { id: "fish", cat: "objects", label: { hy: "Ձկնիկ", en: "Fish", ru: "Рыбка" }, svg: `<svg viewBox="0 0 100 100"><path d="M15,50 C30,25 70,25 85,50 C70,75 30,75 15,50 Z M85,50 L95,65 L90,50 L95,35 Z M30,45 A3,3 0 1,0 30,46" /></svg>`},
        { id: "butterfly", cat: "objects", label: { hy: "Թիթեռ", en: "Butterfly", ru: "Бабочка" }, svg: `<svg viewBox="0 0 100 100"><ellipse cx="50" cy="50" rx="3" ry="25" /><path d="M47,35 C25,10 10,35 47,48 Z" /><path d="M53,35 C75,10 90,35 53,48 Z" /></svg>`},

        // --- NUMBERS (0-9) ---
        { id: "num_0", cat: "numbers", label: { hy: "0", en: "0", ru: "0" }, svg: `<svg viewBox="0 0 100 100"><rect x="30" y="20" width="40" height="60" rx="20" /></svg>`},
        { id: "num_1", cat: "numbers", label: { hy: "1", en: "1", ru: "1" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,30 L50,20 L50,80 M35,80 L65,80" /></svg>`},
        { id: "num_2", cat: "numbers", label: { hy: "2", en: "2", ru: "2" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,35 C30,20 70,20 70,35 C70,55 30,60 30,80 L70,80" /></svg>`},
        { id: "num_3", cat: "numbers", label: { hy: "3", en: "3", ru: "3" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,25 L70,25 L45,50 C55,50 70,58 70,70 C70,82 55,85 40,80" /></svg>`},
        { id: "num_4", cat: "numbers", label: { hy: "4", en: "4", ru: "4" }, svg: `<svg viewBox="0 0 100 100"><path d="M55,15 L30,55 L70,55 M55,15 L55,85" /></svg>`},
        { id: "num_5", cat: "numbers", label: { hy: "5", en: "5", ru: "5" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,20 L35,20 L35,45 C40,40 65,40 65,62 C65,80 45,82 35,75" /></svg>`},
        { id: "num_6", cat: "numbers", label: { hy: "6", en: "6", ru: "6" }, svg: `<svg viewBox="0 0 100 100"><path d="M60,25 C45,25 35,40 35,60 C35,75 45,82 55,82 C65,82 71,72 71,60 C71,48 62,45 52,45" /></svg>`},
        { id: "num_7", cat: "numbers", label: { hy: "7", en: "7", ru: "7" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,22 L70,22 L45,82" /></svg>`},
        { id: "num_8", cat: "numbers", label: { hy: "8", en: "8", ru: "8" }, svg: `<svg viewBox="0 0 100 100"><path d="M50,50 C30,50 30,22 50,22 C70,22 70,50 50,50 Z M50,50 C25,50 25,82 50,82 C75,82 75,50 50,50 Z" /></svg>`},
        { id: "num_9", cat: "numbers", label: { hy: "9", en: "9", ru: "9" }, svg: `<svg viewBox="0 0 100 100"><path d="M48,55 C38,55 29,52 29,40 C29,28 35,18 45,18 C55,18 65,25 65,40 C65,60 55,75 40,82" /></svg>`},

        // --- ARMENIAN LETTERS (Ա-Ֆ) ---
        { id: "hy_A", cat: "hy_letters", label: { hy: "Ա", en: "A", ru: "А" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,75 L35,25 L65,25 L65,75 M35,50 L65,50" /></svg>` },
        { id: "hy_B", cat: "hy_letters", label: { hy: "Բ", en: "B", ru: "Б" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 L65,75 L65,25 Z M35,50 L65,50" /></svg>` },
        { id: "hy_G", cat: "hy_letters", label: { hy: "Գ", en: "G", ru: "Г" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,75 L35,25 L65,25" /></svg>` },
        { id: "hy_D", cat: "hy_letters", label: { hy: "Դ", en: "D", ru: "Д" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,25 L65,75 L35,75 M35,25 L35,45 L65,45" /></svg>` },
        { id: "hy_E", cat: "hy_letters", label: { hy: "Ե", en: "E", ru: "Е" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,25 L35,25 L35,75 L65,75 M35,50 L60,50" /></svg>` },
        { id: "hy_Z", cat: "hy_letters", label: { hy: "Զ", en: "Z", ru: "З" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,25 L35,75 L65,75" /></svg>` },
        { id: "hy_EE", cat: "hy_letters", label: { hy: "Է", en: "EE", ru: "Э" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,25 L35,25 L35,75 M35,50 L60,50" /></svg>` },
        { id: "hy_ET", cat: "hy_letters", label: { hy: "Ը", en: "ET", ru: "Ы" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 M35,25 C55,25 55,50 35,50 C55,50 55,75 35,75" /></svg>` },
        { id: "hy_TO", cat: "hy_letters", label: { hy: "Թ", en: "TO", ru: "Т" }, svg: `<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="25" /><path d="M35,50 L65,50" /></svg>` },
        { id: "hy_JE", cat: "hy_letters", label: { hy: "Ժ", en: "JE", ru: "Ж" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,75 M65,25 L35,75 M50,25 L50,75" /></svg>` },
        { id: "hy_INI", cat: "hy_letters", label: { hy: "Ի", en: "I", ru: "И" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 M65,25 L65,75 M35,75 L65,25" /></svg>` },
        { id: "hy_LYN", cat: "hy_letters", label: { hy: "Լ", en: "L", ru: "Л" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 L65,75" /></svg>` },
        { id: "hy_XAD", cat: "hy_letters", label: { hy: "Խ", en: "X", ru: "Х" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,75 M65,25 L35,75" /></svg>` },
        { id: "hy_CA", cat: "hy_letters", label: { hy: "Ծ", en: "CA", ru: "Ц" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,75 L35,25 C50,25 65,35 65,50 C65,65 50,75 35,75" /></svg>` },
        { id: "hy_KEN", cat: "hy_letters", label: { hy: "Կ", en: "K", ru: "К" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 M35,50 L65,25 M35,50 L65,75" /></svg>` },
        { id: "hy_HO", cat: "hy_letters", label: { hy: "Հ", en: "H", ru: "Х" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 M65,25 L65,75 M35,50 L65,50" /></svg>` },
        { id: "hy_DZА", cat: "hy_letters", label: { hy: "Ձ", en: "DZ", ru: "ДЗ" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,25 L35,50 L65,50 L35,75 L65,75" /></svg>` },
        { id: "hy_GHAT", cat: "hy_letters", label: { hy: "Ղ", en: "GH", ru: "ГХ" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,25 L35,25 L35,75 M35,50 L65,50 L65,75" /></svg>` },
        { id: "hy_TCHE", cat: "hy_letters", label: { hy: "Ճ", en: "TCH", ru: "Ч" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,50 L65,50 L65,75" /></svg>` },
        { id: "hy_MEN", cat: "hy_letters", label: { hy: "Մ", en: "M", ru: "М" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,75 L35,25 L50,50 L65,25 L65,75" /></svg>` },
        { id: "hy_YI", cat: "hy_letters", label: { hy: "Յ", en: "Y", ru: "Й" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,50 L65,50 M50,50 L50,75" /></svg>` },
        { id: "hy_NOW", cat: "hy_letters", label: { hy: "Ն", en: "N", ru: "Н" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 L65,25 L65,75" /></svg>` },
        { id: "hy_SHA", cat: "hy_letters", label: { hy: "Շ", en: "SH", ru: "Ш" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 L65,75 L65,25 M50,25 L50,75" /></svg>` },
        { id: "hy_VO", cat: "hy_letters", label: { hy: "Ո", en: "VO", ru: "О" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 C35,50 65,50 65,25 M50,40 L50,75" /></svg>` },
        { id: "hy_CHA", cat: "hy_letters", label: { hy: "Չ", en: "CH", ru: "Ч" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,50 L65,50 M65,25 L65,75" /></svg>` },
        { id: "hy_PE", cat: "hy_letters", label: { hy: "Պ", en: "P", ru: "П" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,75 L35,25 L65,25 L65,50 L35,50" /></svg>` },
        { id: "hy_DJA", cat: "hy_letters", label: { hy: "Ջ", en: "DJ", ru: "ДЖ" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,25 M50,25 L50,75 L35,75" /></svg>` },
        { id: "hy_RA", cat: "hy_letters", label: { hy: "Ռ", en: "RA", ru: "Р" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,75 L35,25 C55,25 65,35 65,50 C65,65 50,75 65,75" /></svg>` },
        { id: "hy_SE", cat: "hy_letters", label: { hy: "Ս", en: "S", ru: "С" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,35 C65,25 35,25 35,45 C35,65 65,65 65,75 C65,85 35,85 35,75" /></svg>` },
        { id: "hy_VEV", cat: "hy_letters", label: { hy: "Վ", en: "V", ru: "В" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L50,75 L65,25" /></svg>` },
        { id: "hy_TYO", cat: "hy_letters", label: { hy: "Տ", en: "T", ru: "Т" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,25 M50,25 L50,75" /></svg>` },
        { id: "hy_RE", cat: "hy_letters", label: { hy: "Ր", en: "RE", ru: "Р" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,75 M35,25 L65,25 L65,50 L35,50" /></svg>` },
        { id: "hy_TSO", cat: "hy_letters", label: { hy: "Ց", en: "TS", ru: "Ц" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L35,50 L65,25 M35,50 L65,75" /></svg>` },
        { id: "hy_OW", cat: "hy_letters", label: { hy: "Ու", en: "OU", ru: "У" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,25 C25,55 55,55 55,25 M40,45 L40,75 M55,25 L75,75" /></svg>` },
        { id: "hy_PIWR", cat: "hy_letters", label: { hy: "Փ", en: "P", ru: "П" }, svg: `<svg viewBox="0 0 100 100"><path d="M50,15 L50,85 M30,35 C30,15 70,15 70,35 C70,55 30,55 30,35 Z" /></svg>` },
        { id: "hy_KE", cat: "hy_letters", label: { hy: "Ք", en: "K", ru: "К" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 L65,75 M65,25 L35,75 M35,50 L65,50" /></svg>` },
        { id: "hy_EV", cat: "hy_letters", label: { hy: "Եվ", en: "Ev", ru: "Ев" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,45 C45,15 55,45 25,65 L55,65 M65,35 L45,75 L75,75" /></svg>` },
        { id: "hy_O", cat: "hy_letters", label: { hy: "Օ", en: "O", ru: "О" }, svg: `<svg viewBox="0 0 100 100"><ellipse cx="50" cy="50" rx="20" ry="30" /></svg>` },
        { id: "hy_FE", cat: "hy_letters", label: { hy: "Ֆ", en: "F", ru: "Ф" }, svg: `<svg viewBox="0 0 100 100"><path d="M50,15 L50,85 M30,35 C30,15 70,15 70,35 C70,55 30,55 30,35 Z M35,60 L65,60" /></svg>` },

        // --- ENGLISH LETTERS (A-Z) ---
        { id: "en_A", cat: "en_letters", label: { hy: "A", en: "A", ru: "A" }, svg: `<svg viewBox="0 0 100 100"><path d="M50,15 L25,80 M50,15 L75,80 M33,60 L67,60" /></svg>` },
        { id: "en_B", cat: "en_letters", label: { hy: "B", en: "B", ru: "B" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M30,15 L55,15 C65,15 65,48 30,48 M30,48 L60,48 C70,48 70,85 30,85" /></svg>` },
        { id: "en_C", cat: "en_letters", label: { hy: "C", en: "C", ru: "C" }, svg: `<svg viewBox="0 0 100 100"><path d="M70,25 C55,10 30,25 30,50 C30,75 55,90 70,75" /></svg>` },
        { id: "en_D", cat: "en_letters", label: { hy: "D", en: "D", ru: "D" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M30,15 C65,15 75,50 30,85" /></svg>` },
        { id: "en_E", cat: "en_letters", label: { hy: "E", en: "E", ru: "E" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,15 L30,15 L30,85 L65,85 M30,50 L55,50" /></svg>` },
        { id: "en_F", cat: "en_letters", label: { hy: "F", en: "F", ru: "F" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,15 L30,15 L30,85 M30,50 L55,50" /></svg>` },
        { id: "en_G", cat: "en_letters", label: { hy: "G", en: "G", ru: "G" }, svg: `<svg viewBox="0 0 100 100"><path d="M70,25 C55,10 30,25 30,50 C30,75 55,90 70,75 L70,52 L52,52" /></svg>` },
        { id: "en_H", cat: "en_letters", label: { hy: "H", en: "H", ru: "H" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M70,15 L70,85 M30,50 L70,50" /></svg>` },
        { id: "en_I", cat: "en_letters", label: { hy: "I", en: "I", ru: "I" }, svg: `<svg viewBox="0 0 100 100"><path d="M40,15 L60,15 M50,15 L50,85 M40,85 L60,85" /></svg>` },
        { id: "en_J", cat: "en_letters", label: { hy: "J", en: "J", ru: "J" }, svg: `<svg viewBox="0 0 100 100"><path d="M60,15 L60,65 C60,80 35,80 35,65" /></svg>` },
        { id: "en_K", cat: "en_letters", label: { hy: "K", en: "K", ru: "K" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M70,15 L32,50 L70,85" /></svg>` },
        { id: "en_L", cat: "en_letters", label: { hy: "L", en: "L", ru: "L" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,15 L35,85 L70,85" /></svg>` },
        { id: "en_M", cat: "en_letters", label: { hy: "M", en: "M", ru: "M" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,85 L25,15 L50,50 L75,15 L75,85" /></svg>` },
        { id: "en_N", cat: "en_letters", label: { hy: "N", en: "N", ru: "N" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,85 L30,15 L70,85 L70,15" /></svg>` },
        { id: "en_O", cat: "en_letters", label: { hy: "O", en: "O", ru: "O" }, svg: `<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="32" /></svg>` },
        { id: "en_P", cat: "en_letters", label: { hy: "P", en: "P", ru: "P" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M30,15 L55,15 C68,15 68,48 30,48" /></svg>` },
        { id: "en_Q", cat: "en_letters", label: { hy: "Q", en: "Q", ru: "Q" }, svg: `<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="32" /><path d="M62,62 L80,80" /></svg>` },
        { id: "en_R", cat: "en_letters", label: { hy: "R", en: "R", ru: "R" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M30,15 L55,15 C68,15 68,48 30,48 M32,48 L68,85" /></svg>` },
        { id: "en_S", cat: "en_letters", label: { hy: "S", en: "S", ru: "S" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,32 C65,15 35,15 35,35 C35,55 65,50 65,68 C65,85 35,85 35,68" /></svg>` },
        { id: "en_T", cat: "en_letters", label: { hy: "T", en: "T", ru: "T" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L70,15 M50,15 L50,85" /></svg>` },
        { id: "en_U", cat: "en_letters", label: { hy: "U", en: "U", ru: "U" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,60 C30,78 70,78 70,60 L70,15" /></svg>` },
        { id: "en_V", cat: "en_letters", label: { hy: "V", en: "V", ru: "V" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L50,85 L75,15" /></svg>` },
        { id: "en_W", cat: "en_letters", label: { hy: "W", en: "W", ru: "W" }, svg: `<svg viewBox="0 0 100 100"><path d="M20,15 L35,85 L50,40 L65,85 L80,15" /></svg>` },
        { id: "en_X", cat: "en_letters", label: { hy: "X", en: "X", ru: "X" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L75,85 M75,15 L25,85" /></svg>` },
        { id: "en_Y", cat: "en_letters", label: { hy: "Y", en: "Y", ru: "Y" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L50,50 L75,15 M50,50 L50,85" /></svg>` },
        { id: "en_Z", cat: "en_letters", label: { hy: "Z", en: "Z", ru: "Z" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,20 L75,20 L25,80 L75,80" /></svg>` },

        // --- RUSSIAN LETTERS (А-Я) ---
        { id: "ru_A", cat: "ru_letters", label: { hy: "А", en: "A", ru: "А" }, svg: `<svg viewBox="0 0 100 100"><path d="M50,15 L25,80 M50,15 L75,80 M33,60 L67,60" /></svg>` },
        { id: "ru_B", cat: "ru_letters", label: { hy: "Б", en: "B", ru: "Б" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,22 L35,22 L35,78 C35,78 65,78 65,60 C65,45 35,45 35,45" /></svg>`},
        { id: "ru_V", cat: "ru_letters", label: { hy: "В", en: "V", ru: "В" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M30,15 L55,15 C65,15 65,48 30,48 M30,48 L60,48 C70,48 70,85 30,85" /></svg>` },
        { id: "ru_G", cat: "ru_letters", label: { hy: "Г", en: "G", ru: "Г" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,85 L35,15 L70,15" /></svg>` },
        { id: "ru_D", cat: "ru_letters", label: { hy: "Д", en: "D", ru: "Д" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,75 L35,75 L35,25 L65,25 L65,75 L75,75 M25,75 L25,85 M75,75 L75,85" /></svg>` },
        { id: "ru_E", cat: "ru_letters", label: { hy: "Е", en: "E", ru: "Е" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,15 L30,15 L30,85 L65,85 M30,50 L55,50" /></svg>` },
        { id: "ru_YO", cat: "ru_letters", label: { hy: "Ё", en: "YO", ru: "Ё" }, svg: `<svg viewBox="0 0 100 100"><path d="M65,25 L30,25 L30,85 L65,85 M30,55 L55,55 M40,12 A2,2 0 1,0 40,13 M60,12 A2,2 0 1,0 60,13" /></svg>` },
        { id: "ru_ZH", cat: "ru_letters", label: { hy: "Ж", en: "ZH", ru: "Ж" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L70,85 M70,15 L30,85 M50,15 L50,85 M30,50 L70,50" /></svg>` },
        { id: "ru_Z", cat: "ru_letters", label: { hy: "З", en: "Z", ru: "З" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 C35,10 65,10 65,35 C65,50 45,50 45,50 C45,50 65,50 65,70 C65,90 35,90 35,75" /></svg>` },
        { id: "ru_I", cat: "ru_letters", label: { hy: "И", en: "I", ru: "И" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M70,15 L70,85 M70,85 L30,15" /></svg>` },
        { id: "ru_Y", cat: "ru_letters", label: { hy: "Й", en: "Y", ru: "Й" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,25 L30,85 M70,25 L70,85 M70,85 L30,25 M40,15 Q50,8 60,15" /></svg>` },
        { id: "ru_K", cat: "ru_letters", label: { hy: "К", en: "K", ru: "К" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,15 L35,85 M35,50 L65,15 M35,50 L65,85" /></svg>` },
        { id: "ru_L", cat: "ru_letters", label: { hy: "Л", en: "L", ru: "Л" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,85 L45,15 L70,15 L70,85" /></svg>` },
        { id: "ru_M", cat: "ru_letters", label: { hy: "М", en: "M", ru: "М" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,85 L25,15 L50,45 L75,15 L75,85" /></svg>` },
        { id: "ru_N", cat: "ru_letters", label: { hy: "Н", en: "N", ru: "Н" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M70,15 L70,85 M30,50 L70,50" /></svg>` },
        { id: "ru_O", cat: "ru_letters", label: { hy: "О", en: "O", ru: "О" }, svg: `<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="32" /></svg>` },
        { id: "ru_P", cat: "ru_letters", label: { hy: "П", en: "П", ru: "П" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,85 L30,15 L70,15 L70,85" /></svg>` },
        { id: "ru_R", cat: "ru_letters", label: { hy: "Р", en: "R", ru: "Р" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,15 L35,85 M35,15 L60,15 C72,15 72,48 35,48" /></svg>` },
        { id: "ru_S", cat: "ru_letters", label: { hy: "С", en: "С", ru: "С" }, svg: `<svg viewBox="0 0 100 100"><path d="M70,25 C55,10 30,25 30,50 C30,75 55,90 70,75" /></svg>` },
        { id: "ru_T", cat: "ru_letters", label: { hy: "Т", en: "Т", ru: "Т" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L75,15 M50,15 L50,85" /></svg>` },
        { id: "ru_U", cat: "ru_letters", label: { hy: "У", en: "У", ru: "У" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L45,55 L30,85 M45,55 L70,15" /></svg>` },
        { id: "ru_F", cat: "ru_letters", label: { hy: "Ф", en: "F", ru: "Ф" }, svg: `<svg viewBox="0 0 100 100"><path d="M50,15 L50,85 M50,50 C25,50 25,25 50,25 C75,25 75,50 50,50 Z M50,50 C25,50 25,75 50,75 C75,75 75,50 50,50 Z" /></svg>` },
        { id: "ru_X", cat: "ru_letters", label: { hy: "Х", en: "X", ru: "Х" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L75,85 M75,15 L25,85" /></svg>` },
        { id: "ru_TS", cat: "ru_letters", label: { hy: "Ц", en: "TS", ru: "Ц" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,75 L65,75 L65,15 M65,75 L65,85 M65,85 L73,85" /></svg>` },
        { id: "ru_CH", cat: "ru_letters", label: { hy: "Ч", en: "CH", ru: "Ч" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,50 L65,50 M65,15 L65,85" /></svg>` },
        { id: "ru_SH", cat: "ru_letters", label: { hy: "Ш", en: "SH", ru: "Ш" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L25,75 L75,75 L75,15 M42,15 L42,75 M58,15 L58,75" /></svg>` },
        { id: "ru_SHCH", cat: "ru_letters", label: { hy: "Щ", en: "SHCH", ru: "Щ" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,15 L25,75 L75,75 L75,15 M42,15 L42,75 M58,15 L58,75 M75,75 L75,85 L82,85" /></svg>` },
        { id: "ru_HARD", cat: "ru_letters", label: { hy: "Ъ", en: "Ъ", ru: "Ъ" }, svg: `<svg viewBox="0 0 100 100"><path d="M25,20 L45,20 M45,20 L45,80 M45,50 C65,50 65,80 45,80" /></svg>` },
        { id: "ru_YERY", cat: "ru_letters", label: { hy: "Ы", en: "Ы", ru: "Ы" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M30,50 C55,50 55,85 30,85 M65,15 L65,85" /></svg>` },
        { id: "ru_SOFT", cat: "ru_letters", label: { hy: "Ь", en: "Ь", ru: "Ь" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,15 L35,85 M35,50 C60,50 60,85 35,85" /></svg>` },
        { id: "ru_E_REV", cat: "ru_letters", label: { hy: "Э", en: "Э", ru: "Э" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,25 C55,25 60,35 60,50 C60,65 55,75 35,75 M30,50 L55,50" /></svg>` },
        { id: "ru_YU", cat: "ru_letters", label: { hy: "Ю", en: "YU", ru: "Ю" }, svg: `<svg viewBox="0 0 100 100"><path d="M30,15 L30,85 M30,50 L50,50 M75,50 C75,25 50,25 50,50 C50,75 75,75 75,50 Z" /></svg>` },
        { id: "ru_YA", cat: "ru_letters", label: { hy: "Я", en: "YA", ru: "Я" }, svg: `<svg viewBox="0 0 100 100"><path d="M40,85 L40,15 L60,15 C72,15 72,45 40,45 M40,45 L65,85" /></svg>` },

        // --- BIRD LETTERS ---
        { id: "bird_A", cat: "bird_letters", label: { hy: "Ա (Թռչն)", en: "A (Bird)", ru: "Ա (Птиц)" }, svg: `<svg viewBox="0 0 100 100"><path d="M35,75 C25,50 35,25 50,25 C65,25 75,50 65,75" /><path d="M50,25 Q40,10 30,18 Q40,25 50,25" /><path d="M42,50 Q50,45 58,50" /></svg>`}
];


// Helper to get Path2D compatible path
function getPathTemplate(tmplId) {
    const tmpl = GUIDED_TEMPLATES.find(t => t.id === tmplId);
    if (!tmpl) return null;
    if (tmpl.path) return tmpl.path;
    // Extract path from SVG if path field is missing
    const match = tmpl.svg.match(/d="([^"]+)"/);
    return match ? match[1] : "";
}

// Global state for shared use if needed
let currentLanguage = localStorage.getItem('nkarik_lang') || 'hy';
