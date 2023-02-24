
import unittest

import numpy as np

from docutent_distiller.long_document_embedder import BertLongVectorizer
from transformers import logging

logging.set_verbosity_error()


long_text = """2. A felsőoktatás működésének alapelvei  2. § (1) A felsőoktatási intézmény az e törvényben 
meghatározottak szerint - az oktatás, a tudományos kutatás, a művészeti alkotótevékenység mint alaptevékenység 
folytatására - létesített szervezet.  (2) A felsőoktatás rendszerének működtetése az állam, a felsőoktatási intézmény 
működtetése a fenntartó feladata.  (3) A felsőoktatási intézmény oktatási alaptevékenysége magában foglalja a 
felsőoktatási szakképzést, alapképzést, mesterképzést, a doktori képzést és a szakirányú továbbképzést. Az oktatási 
alaptevékenység körébe tartozó tevékenységet - ha e törvény eltérően nem rendelkezik - kizárólag felsőoktatási 
intézmény folytathat.  (4) A felsőoktatási intézmény párt vagy párthoz kötődő szervezet részére helyiségeit működési 
célra nem engedheti át.  (5) Az állam köteles biztosítani, hogy minden képzési területen legyen magyar nyelvű képzés. 
A felsőoktatási intézményben a képzés - részben vagy egészben - nem magyar nyelven is folyhat. A nemzetiséghez 
tartozó hallgató - az e törvényben meghatározottak szerint - anyanyelvén vagy magyar nyelven, illetőleg anyanyelvén 
és magyarul is folytathatja tanulmányait.  (5a) * A felsőoktatási intézmény az alaptevékenységéből származó szellemi 
értékek közösségi célú megismertetésével és gazdasági hasznosításával hozzájárul a térsége társadalmi és gazdasági 
fejlődéséhez.  (6) * A felsőoktatási intézmény a tanulmányi rendszerében a jogszabályban előírt nyilvántartásokat 
köteles vezetni, és abból elektronikus úton köteles adatot szolgáltatni az országos statisztikai adatgyűjtési 
programba, a felsőoktatási információs rendszerbe vagy más, jogszabályban meghatározott rendszerbe.  3. § (1) A 
felsőoktatás egymásra épülő, felsőfokú végzettségi szintet biztosító képzési ciklusai:  a) az alapképzés,  
b) a mesterképzés,  c) a doktori képzés.  (2) * Az alap- és mesterképzést egymásra épülő ciklusokban, 
osztott képzésként, vagy jogszabályban meghatározott esetben osztatlan képzésként lehet megszervezni. A ciklusokra 
bontott, osztott és az osztatlan képzések szerkezetét a felsőoktatásért felelős miniszter (a továbbiakban: miniszter) 
határozza meg.  (3) A felsőoktatás keretében - az (1) bekezdésben foglaltak mellett - felsőfokú végzettségi szintet 
nem biztosító képzésként  a) felsőoktatási szakképzés,  b) szakirányú továbbképzés  is szervezhető.  (4) * A 
felsőoktatási intézmények az alapító okiratukban foglaltak alapján - a felnőttképzésről szóló törvény szerint - 
vehetnek részt a felnőttképzésben.  4. § (1) Felsőoktatási intézményt önállóan vagy más jogosulttal együttesen  a) * 
a magyar állam, országos nemzetiségi önkormányzat,  b) * az egyházi jogi személy (a továbbiakban: egyházi fenntartó), 
 c) * a Magyarország területén székhellyel rendelkező gazdasági társaság,  d) * a Magyarországon nyilvántartásba vett 
 alapítvány, vagyonkezelő alapítvány, közalapítvány vagy vallási egyesület  alapíthat.  (1a) * Egyházi felsőoktatási 
 intézmény az (1) bekezdés b) pontja szerinti fenntartó által fenntartott felsőoktatási intézmény. Magán 
 felsőoktatási intézmény az (1) bekezdés c) és d) pontja szerinti fenntartó által fenntartott felsőoktatási 
 intézmény.  (2) Az alapítói jogok gyakorlásának joga az e törvényben meghatározottak szerint átruházható. Az, 
 aki az alapítói jogot gyakorolja, ellátja a felsőoktatási intézmény fenntartásával kapcsolatos feladatokat (a 
 továbbiakban: fenntartó).  (2a) * Az állami fenntartó kivételével az (1) bekezdés szerinti fenntartó által 
 fenntartott intézmény esetében - eltérő megállapodás hiányában - tulajdonos az, aki az alapítói, illetve fenntartói 
 jogot gyakorolja. A tulajdonos a fenntartói jogot e törvényben meghatározottak szerint ruházhatja át az (1) bekezdés 
 szerinti jogosultnak. Tulajdonos csak az lehet, aki az (1) bekezdés alapján felsőoktatási intézményt alapíthat.  (
 2b) * Ha a felek megállapodása alapján a tulajdonos és a fenntartói jog gyakorlója eltér, az erről szóló bejelentés 
 alapján a felsőoktatási intézmények nyilvántartását vezető szerv (a továbbiakban: oktatási hivatal) a tulajdonost és 
 a fenntartói jog gyakorlóját is nyilvántartásba veszi.  (3) Költségvetési szervként működik a felsőoktatási 
 intézmény, ha az (1) bekezdés a) pontjában meghatározottak tartják fenn. Az (1) bekezdés a) pontjában felsoroltak 
 közösen, illetve az (1) bekezdés b)-d) pontjában meghatározottak közösen is gyakorolhatják a fenntartói jogokat.  (
 4) * Az állam nevében a fenntartói jogokat - ha törvény másként nem rendelkezik - a miniszter gyakorolja. A 
 miniszter a fenntartói jogot megállapodással a tudománypolitika koordinációjáért felelős miniszterre ruházhatja.  (
 5) * A (2a) és (2b) bekezdés szerinti esetben  a) a tulajdonos - a felsőoktatási intézménynek a kutatás és az 
 oktatás tartalmával és módszereivel kapcsolatban, az Alaptörvényben és az e törvényben biztosított önállóságát nem 
 sértve - gyakorolja a tulajdonost a polgári jog alapján megillető jogokat, azokat az (1) bekezdésben meghatározottak 
 részére polgári jogi megállapodással vagy egyoldalú jognyilatkozattal ruházhatja át,  b) a 73. § (1) bekezdése 
 szerinti fenntartói irányítást a tulajdonos vagy - a felek eltérő megállapodása esetén - az oktatási hivatal 
 nyilvántartásába bejegyzett fenntartó gyakorolja.  5. § (1) * A felsőoktatási intézmény, valamint a felsőoktatási 
 intézmény 94. § (2c) bekezdésben meghatározott szervezeti egysége jogi személy.  (2) * A munka törvénykönyvét, 
 valamint - az állami felsőoktatási intézmény tekintetében - a közalkalmazottak jogállásáról szóló törvényt e 
 törvényben meghatározott eltérésekkel kell alkalmazni.  (3) A felsőoktatási intézmény e törvény szerinti átalakulása 
 - egyesülése, kiválása, beolvadása - nem tartozik a tisztességtelen piaci magatartás és a versenykorlátozás 
 tilalmáról szóló törvény szerinti piaci magatartás körébe.    6. § (1) Felsőoktatási intézményként olyan szervezet 
 hozható létre illetve működhet, amelyet az e törvényben meghatározott felsőoktatási feladatok ellátására 
 létesítettek és az Országgyűléstől megkapta az állami elismerést. (2) Állami elismerést az a felsőoktatási intézmény 
 kaphat, amelyik rendelkezik a feladatai ellátásához szükséges feltételekkel, és az a)-d) pontok szerint választható 
 képzési szerkezetben, legalább két képzési, illetve tudományterületen legalább négy szakon a) alapképzést, 
 b) alap- és mesterképzést, c) alap-, valamint mester- és doktori képzést, d) mester- és doktori képzést jogosult 
 folytatni. (3) A felsőoktatási intézmény akkor rendelkezik a feladatai ellátásához szükséges feltételekkel, 
 ha - az alapító okiratában meghatározott feladatai figyelembevételével - a folyamatos működéséhez szükséges 
 személyi, szervezeti feltételek, tárgyi és pénzügyi eszközök, valamint az intézményi dokumentumok rendelkezésére 
 állnak. (4) A felsőoktatási intézmény állami elismeréssel jön létre. (5) A felsőoktatási intézmény a működését akkor 
 kezdheti meg, ha a) * a fenntartó kérelmére az oktatási hivataltól megkapta a működési engedélyt, nyilvántartásba 
 vették és b) az Országgyűlés döntött az állami elismeréséről."""

short_text = "Teszt mondat."
class LongVectorizerTestCase(unittest.TestCase):


    def test_initialize(self):
        vectorizer = BertLongVectorizer()
    def test_segmentation_empty(self):
        vectorizer = BertLongVectorizer()
        result = vectorizer.vectorize("")
        expected = np.array([0])
        self.assertTrue(np.equal(result, expected))

    def test_segmentation_short_matrix(self):
        vectorizer = BertLongVectorizer()
        matrix = vectorizer.vectorize(short_text, matrix=True)
        self.assertEqual(matrix.shape, (768,))

    def test_segmentation_short_mean(self):
        vectorizer = BertLongVectorizer()
        mean_vector = vectorizer.vectorize(short_text, matrix=False)
        self.assertEqual(mean_vector.shape, (768,))

    def test_segmentation_short_class_variables(self):
        vectorizer = BertLongVectorizer()
        _ = vectorizer.vectorize(short_text, matrix=True)
        self.assertEqual(len(vectorizer.connected_sw_tokens), 3)
        self.assertEqual(vectorizer.slicing_points[0], 2)
        self.assertEqual(len(vectorizer.slices), 1)

    def test_segmentation_long_matrix(self):
        vectorizer = BertLongVectorizer()
        matrix = vectorizer.vectorize(long_text, matrix=True)
        self.assertEqual(matrix.shape, (3, 768))

    def test_segmentation_long_mean(self):
        vectorizer = BertLongVectorizer()
        mean_vector = vectorizer.vectorize(long_text, matrix=False)
        self.assertEqual(mean_vector.shape, (768,))

    def test_segmentation_long_class_variables(self):
        vectorizer = BertLongVectorizer()
        _ = vectorizer.vectorize(long_text, matrix=True)
        self.assertEqual(len(vectorizer.connected_sw_tokens), 1276)
        self.assertEqual(len(vectorizer.slices), 3)
        self.assertEqual(vectorizer.slicing_points, [509, 1018, 1275])


if __name__ == "__main__":
    unittest.main()